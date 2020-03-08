from collections import namedtuple
from itertools import chain
from abc import ABC, abstractmethod
import os
from typing import List, Any

import numpy as np
import mip.model as mip
from model.constants import Constants
from model.visualizer import Visualizer


class SchemaSetsLearner:
    def __init__(self,
            replay_buffer, schema_len=Constants.SCHEMA_VEC_SIZE, n_targets=Constants.N_PREDICTABLE_ATTRIBUTES
    ):
        self._replay_buffer = ReplayBuffer(n_features=schema_len, n_targets=n_targets)

        self._attr_schema_sets = [SchemaSet(schema_len) for _ in range(n_targets)]
        self._rew_schema_set = SchemaSet(schema_len)

        self._attr_mip_models = [PythonMipModelWrapper(n_vars=schema_len) for _ in range(self.N_PREDICTABLE_ATTRIBUTES)]
        self._reward_mip_model = PythonMipModelWrapper(n_vars=schema_len)

        self._solved = []
        self._curr_iter = None
        self._visualizer = Visualizer(None, None, None)

    def set_curr_iter(self, curr_iter):
        self._curr_iter = curr_iter
        self._visualizer.set_iter(curr_iter)

    def take_batch(self, batch):
        self._replay_buffer.append_to_new_batch(batch)

    def _add_to_replay_and_constraints_buff(self, batch):
        batch = self._replay_buffer._flush_new_batch()
        if batch is None:
            return

        self._replay_buffer.add_new_batch_to_replay_buffer(batch)

        # === UPDATE Model Constraints ===

        # # find non-duplicate indices in new batch (batch-based indexing)
        # new_batch_mask = unique_idx >= replay_size
        # new_non_duplicate_indices = unique_idx[new_batch_mask] - replay_size
        #
        # # find indices that will index constraints_buff + new_batch_unique synchronously with replay
        # constraints_unique_idx = unique_idx.copy()
        # constraints_unique_idx[new_batch_mask] = replay_size + np.arange(len(new_non_duplicate_indices))
        #
        # for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
        #     attr_batch = (batch.x[new_non_duplicate_indices],
        #                   batch.y[new_non_duplicate_indices, attr_idx])
        #     self._attr_mip_models[attr_idx].add_to_constraints_buff(attr_batch, constraints_unique_idx)
        #
        # reward_batch = (batch.x[new_non_duplicate_indices],
        #                 batch.r[new_non_duplicate_indices])
        # self._reward_mip_model.add_to_constraints_buff(reward_batch, constraints_unique_idx,
        #                                                replay_renewed_indices=replay_renewed_indices)


    def _purge_matrix_columns(self, matrix, col_indices):
        n_cols_purged = len(col_indices)

        if n_cols_purged:
            col_size, _ = matrix.shape
            matrix = np.delete(matrix, col_indices, axis=1)
            padding = np.ones((col_size, n_cols_purged), dtype=bool)
            matrix = np.hstack((matrix, padding))

        return matrix, n_cols_purged

    def _delete_attr_schema_vectors(self, attr_idx, vec_indices):
        matrix, n_cols_purged = self._purge_matrix_columns(self._W[attr_idx], vec_indices)
        if n_cols_purged:
            self._W[attr_idx] = matrix
            self._n_attr_schemas[attr_idx] -= n_cols_purged
        return n_cols_purged

    def _delete_reward_schema_vectors(self, vec_indices):
        matrix, n_cols_purged = self._purge_matrix_columns(self._R, vec_indices)
        if n_cols_purged:
            self._R = matrix
            self._n_reward_schemas -= n_cols_purged
        return n_cols_purged

    def _delete_incorrect_schemas(self, batch):
        augmented_entities, targets, rewards = batch
        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            self._delete_incorrect_attr_schema(attr_idx, augmented_entities, targets)

        self._delete_incorrect_reward_schema(augmented_entities, rewards)

    def _delete_incorrect_reward_schema(self, augmented_entities, rewards):
        reward_prediction = self._rew_schema_set.predict(augmented_entities)

        # false positive predictions
        mispredicted_samples_mask = reward_prediction.any(axis=1) & ~rewards
        incorrect_schemas_mask = reward_prediction[mispredicted_samples_mask, :].any(axis=0)
        incorrect_schemas_indices = np.nonzero(incorrect_schemas_mask)[0]
        assert incorrect_schemas_indices.ndim == 1
        n_schemas_deleted = self._delete_reward_schema_vectors(incorrect_schemas_indices)
        if n_schemas_deleted:
            print('Deleted incorrect reward schemas: {}'.format(n_schemas_deleted))

    def _delete_incorrect_attr_schema(self, attr_idx, augmented_entities, targets):
        attr_prediction = self._attr_schema_sets[attr_idx].predict(augmented_entities)

        # false positive predictions
        mispredicted_samples_mask = attr_prediction.any(axis=1) & ~targets[:, attr_idx]
        incorrect_schemas_mask = attr_prediction[mispredicted_samples_mask, :].any(axis=0)
        incorrect_schemas_indices = np.nonzero(incorrect_schemas_mask)[0]
        assert incorrect_schemas_indices.ndim == 1

        self._attr_schema_sets[attr_idx].remove_schemas(incorrect_schemas_indices)
        if n_schemas_deleted:
            print('Deleted incorrect attr schemas: {} of {}'.format(
                len(incorrect_schemas_indices), self.ENTITY_NAMES[attr_idx]))

    def _find_cluster(self, zp_pl_mask, zp_nl_mask, augmented_entities, target, attr_idx, is_reward=False):
        """
        augmented_entities: zero-predicted only
        target: scalar vector
        """
        assert augmented_entities.dtype == np.int
        assert target.dtype == np.int

        # find all entries, that can be potentially solved (have True labels)
        candidates = augmented_entities[zp_pl_mask]

        print('finding cluster...')
        print('augmented_entities: {}'.format(augmented_entities.shape[0]))
        print('zp pos samples: {}'.format(candidates.shape[0]))

        if not candidates.size:
            return None

        zp_pl_indices = np.nonzero(zp_pl_mask)[0]

        #if not is_reward:
        # sample one entry and add it's idx to 'solved'
        idx = np.random.choice(zp_pl_indices)
        self._solved.append(idx)

        # resample candidates
        zp_pl_mask[idx] = False
        zp_pl_indices = np.nonzero(zp_pl_mask)[0]
        candidates = augmented_entities[zp_pl_mask]

        # solve LP
        objective_coefficients = (1 - candidates).sum(axis=0)
        objective_coefficients = list(objective_coefficients)

        if not is_reward:
            new_schema_vector = self._attr_mip_models[attr_idx].optimize(objective_coefficients, zp_nl_mask,
                                                                         self._solved)
        else:
            new_schema_vector = self._reward_mip_model.optimize(objective_coefficients, zp_nl_mask, self._solved)

        if new_schema_vector is None:
            print('!!! Cannot find cluster !!!')
            return None

        # add all samples that are solved by just learned schema vector
        if candidates.size:
            new_predicted_attribute = (1 - candidates) @ new_schema_vector
            cluster_members_mask = np.isclose(new_predicted_attribute, 0, rtol=0, atol=self.ADDING_SCHEMA_TOLERANCE)
            n_new_members = np.count_nonzero(cluster_members_mask)

            if n_new_members:
                print('Also added to solved: {}'.format(n_new_members))
                self._solved.extend(zp_pl_indices[cluster_members_mask])
            #elif is_reward:
            #    # constraints are satisfied but new vector does not predict any r = 1
            #    new_schema_vector = None
            #    print('TRASH REWARD SCHEMA DISCARDED')

        return new_schema_vector

    def _simplify_schema(self, zp_nl_mask, schema_vector, augmented_entities, target, attr_idx, is_reward=False):
        objective_coefficients = [1] * len(schema_vector)

        if not is_reward:
            model = self._attr_mip_models[attr_idx]
        else:
            model = self._reward_mip_model

        new_schema_vector = model.optimize(objective_coefficients, zp_nl_mask, self._solved)
        assert new_schema_vector is not None

        return new_schema_vector


    def _generate_new_schema(self, augmented_entities, targets, attr_idx, is_reward=False):
        if not is_reward:
            target = targets[:, attr_idx].astype(np.int, copy=False)
            prediction = self._attr_schema_sets[attr_idx].predict(augmented_entities)
        else:
            target = targets.astype(np.int, copy=False)
            prediction = self._rew_schema_set.predict(augmented_entities)

        augmented_entities = augmented_entities.astype(np.int, copy=False)

        # sample only entries with zero-prediction
        zp_mask = ~prediction.any(axis=1)
        pl_mask = target == 1
        # pos and neg labels' masks
        zp_pl_mask = zp_mask & pl_mask
        zp_nl_mask = zp_mask & ~pl_mask

        new_schema_vector = self._find_cluster(zp_pl_mask, zp_nl_mask,
                                               augmented_entities, target, attr_idx,
                                               is_reward=is_reward)
        if new_schema_vector is None:
            return None

        new_schema_vector = self._simplify_schema(zp_nl_mask, new_schema_vector,
                                                  augmented_entities, target, attr_idx,
                                                  is_reward=is_reward)
        if new_schema_vector is None:
            print('!!! Cannot simplify !!!')
            return None

        new_schema_vector = self._binarize_schema(new_schema_vector)

        self._solved.clear()

        return new_schema_vector

    @property
    def learned_W(self):
        return [schema_set.W for schema_set in self._attr_schema_sets]


    @property
    def learned_R(self):
        return [self._rew_schema_set.W]

    def learn(self):
        # get full batch from buffer
        new_batch = self._replay_buffer._flush_new_batch()
        if buff_batch is not None:
            self._delete_incorrect_schemas(new_batch)
            self._add_to_replay_and_constraints_buff(new_batch)

        # get all data to learn on
        replay_batch = self._replay_buffer.replay_batch
        if replay_batch is None:
            return

        augmented_entities, targets, rewards = replay_batch

        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            while self._n_attr_schemas[attr_idx] < self.L:
                new_schema_vec = self._generate_new_schema(augmented_entities, targets, attr_idx)
                if new_schema_vec is None:
                    break
                self._attr_schema_sets[attr_idx].add_schema(schema=new_schema_vec)

        while self._n_reward_schemas < self.L:
            new_schema_vec = self._generate_new_schema(augmented_entities, rewards, None, is_reward=True)
            if new_schema_vec is None:
                break
            self._rew_schema_set.add_schema(new_schema_vec)

        self._visualizer.visualize_and_dump_schemas(self.learned_W, self.learned_R)
        self._visualizer.visualize_replay_buffer(self._replay_buffer.replay_batch)



class GreedySchemaLearner(Constants):
    Batch = namedtuple('Batch', ['x', 'y', 'r'])

    def __init__(self):
        self._W = [np.ones((self.SCHEMA_VEC_SIZE, self.L), dtype=bool)
                   for _ in range(self.N_PREDICTABLE_ATTRIBUTES)]
        self._n_attr_schemas = np.ones(self.N_PREDICTABLE_ATTRIBUTES, dtype=np.int)

        self._R = np.ones((self.SCHEMA_VEC_SIZE, self.L), dtype=bool)
        self._n_reward_schemas = 1

        self._buff = []
        self._replay = self.Batch(np.empty((0, self.SCHEMA_VEC_SIZE), dtype=bool),
                                  np.empty((0, self.N_PREDICTABLE_ATTRIBUTES), dtype=bool),
                                  np.empty((0), dtype=bool))

        self._attr_mip_models = [MipModel() for _ in range(self.N_PREDICTABLE_ATTRIBUTES)]
        self._reward_mip_model = MipModel()
        self._solved = []

        self._curr_iter = None
        self._visualizer = Visualizer(None, None, None)

    def set_curr_iter(self, curr_iter):
        self._curr_iter = curr_iter
        self._visualizer.set_iter(curr_iter)

    def take_batch(self, batch):
        for part in batch:
            assert part.dtype == bool

        if batch.x.size:
            x, y, r = self._handle_duplicates(batch.x, batch.y, batch.r)
            filtered_batch = self.Batch(x, y, r)
            self._buff.append(filtered_batch)

    def _handle_duplicates(self, augmented_entities, target, rewards, return_index=False):
        keys = np.packbits(augmented_entities, axis=1)
        samples, idx = np.unique(keys, axis=0, return_index=True)
        out = [augmented_entities[idx], target[idx], rewards[idx]]
        if return_index:
            out.append(idx)
        return tuple(out)

    def _get_buff_batch(self):
        out = None
        if self._buff:
            # sort buff to keep r = 0 entries
            self._buff = sorted(self._buff, key=lambda batch: batch.r[0])

            x, y, r = zip(*self._buff)
            augmented_entities = np.concatenate(x, axis=0)
            targets = np.concatenate(y, axis=0)
            rewards = np.concatenate(r, axis=0)

            augmented_entities, targets, rewards = self._handle_duplicates(augmented_entities, targets, rewards)
            out = self.Batch(augmented_entities, targets, rewards)
        return out

    def _add_to_replay_and_constraints_buff(self, batch):
        replay_size = len(self._replay.x)

        # concatenate replay + batch
        x = np.concatenate((self._replay.x, batch.x), axis=0)
        y = np.concatenate((self._replay.y, batch.y), axis=0)
        r = np.concatenate((self._replay.r, batch.r), axis=0)

        # remove duplicates
        x_filtered, y_filtered, r_filtered, unique_idx = self._handle_duplicates(x, y, r, return_index=True)
        self._replay = self.Batch(x_filtered, y_filtered, r_filtered)

        # find r = 0 duplicates (they can only locate in batch)
        batch_size = len(batch.x)
        concat_size = len(x)

        duplicates_mask = np.ones(concat_size, dtype=bool)
        duplicates_mask[unique_idx] = False
        no_reward_mask = r == 0
        reward_renew_indices = np.nonzero(duplicates_mask & no_reward_mask)[0]
        assert (reward_renew_indices >= replay_size).all()
        reward_renew_samples = x[reward_renew_indices]

        # renew rewards to zero
        replay_renewed_indices = []
        for sample in reward_renew_samples:
            indices = np.nonzero((self._replay.x == sample).all(axis=1))[0]
            assert len(indices) == 1
            idx = indices[0]
            if self._replay.r[idx] != 0:
                self._replay.r[idx] = 0
                replay_renewed_indices.append(idx)
        print('new r=0 samples overwritten: {}'.format(reward_renew_indices.size))

        # find non-duplicate indices in new batch (batch-based indexing)
        new_batch_mask = unique_idx >= replay_size
        new_non_duplicate_indices = unique_idx[new_batch_mask] - replay_size

        # find indices that will index constraints_buff + new_batch_unique synchronously with replay
        constraints_unique_idx = unique_idx.copy()
        constraints_unique_idx[new_batch_mask] = replay_size + np.arange(len(new_non_duplicate_indices))

        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            attr_batch = (batch.x[new_non_duplicate_indices],
                          batch.y[new_non_duplicate_indices, attr_idx])
            self._attr_mip_models[attr_idx].add_to_constraints_buff(attr_batch, constraints_unique_idx)

        reward_batch = (batch.x[new_non_duplicate_indices],
                        batch.r[new_non_duplicate_indices])
        self._reward_mip_model.add_to_constraints_buff(reward_batch, constraints_unique_idx,
                                                       replay_renewed_indices=replay_renewed_indices)

    def _get_replay_batch(self):
        if self._replay.x.size:
            out = self._replay
        else:
            out = None
        return out

    def _predict_attribute(self, augmented_entities, attr_idx):
        assert augmented_entities.dtype == bool

        n_schemas = self._n_attr_schemas[attr_idx]
        W = self._W[attr_idx][:, :n_schemas]
        attribute_prediction = ~(~augmented_entities @ W)
        return attribute_prediction

    def _predict_reward(self, augmented_entities):
        assert augmented_entities.dtype == bool

        R = self._R[:, :self._n_reward_schemas]
        reward_prediction = ~(~augmented_entities @ R)
        return reward_prediction

    def _add_attr_schema_vec(self, attr_idx, schema_vec):
        vec_idx = self._n_attr_schemas[attr_idx]
        if vec_idx < self.L:
            self._W[attr_idx][:, vec_idx] = schema_vec
            self._n_attr_schemas[attr_idx] += 1

    def _add_reward_schema_vec(self, schema_vec):
        vec_idx = self._n_reward_schemas
        if vec_idx < self.L:
            self._R[:, vec_idx] = schema_vec
            self._n_reward_schemas += 1

    def _purge_matrix_columns(self, matrix, col_indices):
        n_cols_purged = len(col_indices)

        if n_cols_purged:
            col_size, _ = matrix.shape
            matrix = np.delete(matrix, col_indices, axis=1)
            padding = np.ones((col_size, n_cols_purged), dtype=bool)
            matrix = np.hstack((matrix, padding))

        return matrix, n_cols_purged

    def _delete_attr_schema_vectors(self, attr_idx, vec_indices):
        matrix, n_cols_purged = self._purge_matrix_columns(self._W[attr_idx], vec_indices)
        if n_cols_purged:
            self._W[attr_idx] = matrix
            self._n_attr_schemas[attr_idx] -= n_cols_purged
        return n_cols_purged

    def _delete_reward_schema_vectors(self, vec_indices):
        matrix, n_cols_purged = self._purge_matrix_columns(self._R, vec_indices)
        if n_cols_purged:
            self._R = matrix
            self._n_reward_schemas -= n_cols_purged
        return n_cols_purged

    def _delete_incorrect_schemas(self, batch):
        augmented_entities, targets, rewards = batch
        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            self._delete_incorrect_attr_schema(attr_idx, augmented_entities, targets)

        self._delete_incorrect_reward_schema(augmented_entities, rewards)

    def _delete_incorrect_reward_schema(self, augmented_entities, rewards):
        reward_prediction = self._predict_reward(augmented_entities)
        # false positive predictions
        mispredicted_samples_mask = reward_prediction.any(axis=1) & ~rewards
        incorrect_schemas_mask = reward_prediction[mispredicted_samples_mask, :].any(axis=0)
        incorrect_schemas_indices = np.nonzero(incorrect_schemas_mask)[0]
        assert incorrect_schemas_indices.ndim == 1
        n_schemas_deleted = self._delete_reward_schema_vectors(incorrect_schemas_indices)
        if n_schemas_deleted:
            print('Deleted incorrect reward schemas: {}'.format(n_schemas_deleted))

    def _delete_incorrect_attr_schema(self, attr_idx, augmented_entities, targets):
        attr_prediction = self._predict_attribute(augmented_entities, attr_idx)
        # false positive predictions
        mispredicted_samples_mask = attr_prediction.any(axis=1) & ~targets[:, attr_idx]
        incorrect_schemas_mask = attr_prediction[mispredicted_samples_mask, :].any(axis=0)
        incorrect_schemas_indices = np.nonzero(incorrect_schemas_mask)[0]
        assert incorrect_schemas_indices.ndim == 1
        n_schemas_deleted = self._delete_attr_schema_vectors(attr_idx, incorrect_schemas_indices)
        if n_schemas_deleted:
            print('Deleted incorrect attr schemas: {} of {}'.format(
                n_schemas_deleted, self.ENTITY_NAMES[attr_idx]))

    def _find_cluster(self, zp_pl_mask, zp_nl_mask, augmented_entities, target, attr_idx, is_reward=False):
        """
        augmented_entities: zero-predicted only
        target: scalar vector
        """
        assert augmented_entities.dtype == np.int
        assert target.dtype == np.int

        # find all entries, that can be potentially solved (have True labels)
        candidates = augmented_entities[zp_pl_mask]

        print('finding cluster...')
        print('augmented_entities: {}'.format(augmented_entities.shape[0]))
        print('zp pos samples: {}'.format(candidates.shape[0]))

        if not candidates.size:
            return None

        zp_pl_indices = np.nonzero(zp_pl_mask)[0]

        #if not is_reward:
        # sample one entry and add it's idx to 'solved'
        idx = np.random.choice(zp_pl_indices)
        self._solved.append(idx)

        # resample candidates
        zp_pl_mask[idx] = False
        zp_pl_indices = np.nonzero(zp_pl_mask)[0]
        candidates = augmented_entities[zp_pl_mask]

        # solve LP
        objective_coefficients = (1 - candidates).sum(axis=0)
        objective_coefficients = list(objective_coefficients)

        if not is_reward:
            new_schema_vector = self._attr_mip_models[attr_idx].optimize(objective_coefficients, zp_nl_mask,
                                                                         self._solved)
        else:
            new_schema_vector = self._reward_mip_model.optimize(objective_coefficients, zp_nl_mask, self._solved)

        if new_schema_vector is None:
            print('!!! Cannot find cluster !!!')
            return None

        # add all samples that are solved by just learned schema vector
        if candidates.size:
            new_predicted_attribute = (1 - candidates) @ new_schema_vector
            cluster_members_mask = np.isclose(new_predicted_attribute, 0, rtol=0, atol=self.ADDING_SCHEMA_TOLERANCE)
            n_new_members = np.count_nonzero(cluster_members_mask)

            if n_new_members:
                print('Also added to solved: {}'.format(n_new_members))
                self._solved.extend(zp_pl_indices[cluster_members_mask])
            #elif is_reward:
            #    # constraints are satisfied but new vector does not predict any r = 1
            #    new_schema_vector = None
            #    print('TRASH REWARD SCHEMA DISCARDED')

        return new_schema_vector

    def _simplify_schema(self, zp_nl_mask, schema_vector, augmented_entities, target, attr_idx, is_reward=False):
        objective_coefficients = [1] * len(schema_vector)

        if not is_reward:
            model = self._attr_mip_models[attr_idx]
        else:
            model = self._reward_mip_model

        new_schema_vector = model.optimize(objective_coefficients, zp_nl_mask, self._solved)
        assert new_schema_vector is not None

        return new_schema_vector

    def _binarize_schema(self, schema_vector):
        threshold = 0.5
        return schema_vector > threshold

    def _generate_new_schema(self, augmented_entities, targets, attr_idx, is_reward=False):
        if not is_reward:
            target = targets[:, attr_idx].astype(np.int, copy=False)
            prediction = self._predict_attribute(augmented_entities, attr_idx)
        else:
            target = targets.astype(np.int, copy=False)
            prediction = self._predict_reward(augmented_entities)

        augmented_entities = augmented_entities.astype(np.int, copy=False)

        # sample only entries with zero-prediction
        zp_mask = ~prediction.any(axis=1)
        pl_mask = target == 1
        # pos and neg labels' masks
        zp_pl_mask = zp_mask & pl_mask
        zp_nl_mask = zp_mask & ~pl_mask

        new_schema_vector = self._find_cluster(zp_pl_mask, zp_nl_mask,
                                               augmented_entities, target, attr_idx,
                                               is_reward=is_reward)
        if new_schema_vector is None:
            return None

        new_schema_vector = self._simplify_schema(zp_nl_mask, new_schema_vector,
                                                  augmented_entities, target, attr_idx,
                                                  is_reward=is_reward)
        if new_schema_vector is None:
            print('!!! Cannot simplify !!!')
            return None

        new_schema_vector = self._binarize_schema(new_schema_vector)

        self._solved.clear()

        return new_schema_vector

    def dump_weights(self, learned_W, learned_R):
        dir_name = 'dump'
        os.makedirs(dir_name, exist_ok=True)
        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            file_name = 'w_{}.pkl'.format(attr_idx)
            path = os.path.join(dir_name, file_name)
            learned_W[attr_idx].dump(path)

        file_name = 'r_pos.pkl'
        path = os.path.join(dir_name, file_name)
        learned_R[0].dump(path)

    def get_weights(self):
        learned_W = [W[:, ~np.all(W, axis=0)] for W in self._W]
        if any(w.size == 0 for w in learned_W):
            learned_W = None

        learned_R = [self._R[:, ~np.all(self._R, axis=0)]]
        return learned_W, learned_R

    def learn(self):
        # get full batch from buffer
        buff_batch = self._get_buff_batch()
        if buff_batch is not None:
            self._delete_incorrect_schemas(buff_batch)
            self._add_to_replay_and_constraints_buff(buff_batch)

        # get all data to learn on
        replay_batch = self._get_replay_batch()
        if replay_batch is None:
            return

        augmented_entities, targets, rewards = replay_batch

        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            while self._n_attr_schemas[attr_idx] < self.L:
                new_schema_vec = self._generate_new_schema(augmented_entities, targets, attr_idx)
                if new_schema_vec is None:
                    break
                self._add_attr_schema_vec(attr_idx, new_schema_vec)

        while self._n_reward_schemas < self.L:
            new_schema_vec = self._generate_new_schema(augmented_entities, rewards, None, is_reward=True)
            if new_schema_vec is None:
                break
            self._add_reward_schema_vec(new_schema_vec)

        if self.VISUALIZE_SCHEMAS:
            learned_W = [W[:, ~np.all(W, axis=0)] for W in self._W]
            learned_R = [self._R[:, ~np.all(self._R, axis=0)]]
            self._visualizer.visualize_schemas(learned_W, learned_R)
            self.dump_weights(learned_W, learned_R)

        if self.VISUALIZE_REPLAY_BUFFER:
            self._visualizer.visualize_replay_buffer(self._replay)



ExperienceBatch = namedtuple('ExperienceBatch', ['x', 'y', 'r'])


class ReplayBuffer:
    _new_batch_buffer: List[ExperienceBatch]
    _replay_batch: ExperienceBatch

    def __init__(self, n_features=Constants.SCHEMA_VEC_SIZE, n_targets=Constants.N_PREDICTABLE_ATTRIBUTES):
        self._new_batch_buffer = []
        self._replay_batch = ExperienceBatch(
            np.empty((0, n_features), dtype=bool),
            np.empty((0, n_targets), dtype=bool),
            np.empty((0), dtype=bool)
        )

    def sync_replay_buffer(self):
        batch = self._flush_new_batch()
        added_indices, changed_reward_indices = self._get_added_and_changed(batch)

        # added indices relative to the new replay
        old_len = self._replay_batch.x.shape[0]
        new_indices = list(range(old_len, old_len + len(added_indices)))

        # make new replay batch
        x = np.concatenate((self._replay_batch.x, batch.x[added_indices]), axis=0)
        y = np.concatenate((self._replay_batch.y, batch.y[added_indices]), axis=0)
        r = np.concatenate((self._replay_batch.r, batch.r[added_indices]), axis=0)
        self._replay_batch = ExperienceBatch(x, y, r)

        # update rewards
        self._replay_batch.r[changed_reward_indices] = 0
        print('new r=0 samples overwritten: {}'.format(len(changed_reward_indices)))

        # both indices now replay-wise
        return new_indices, changed_reward_indices

    @property
    def replay_batch(self) -> ExperienceBatch:
        return self._replay_batch

    def append_to_new_batch(self, batch):
        for part in batch:
            assert part.dtype == bool

        x, y, r = batch
        if x.size:
            return

        x, y, r = self._get_uniques(x, y, r)
        filtered_batch = ExperienceBatch(x, y, r)
        self._new_batch_buffer.append(filtered_batch)

    def _get_uniques(self, x, y, r, return_index=False):
        keys = np.packbits(x, axis=1)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)

        result = (x[unique_indices], y[unique_indices], r[unique_indices])
        if return_index:
            return result + (unique_indices, )
        return result

    def _flush_new_batch(self):
        assert self._new_batch_buffer

        # sort it to keep r = 0 entries
        self._new_batch_buffer = sorted(self._new_batch_buffer, key=lambda batch: batch.r[0])

        x, y, r = zip(*self._new_batch_buffer)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        r = np.concatenate(r, axis=0)

        self._new_batch_buffer = []

        x, y, r = self._get_uniques(x, y, r)
        return ExperienceBatch(x, y, r)

    @staticmethod
    def _find_row_index(row: np.ndarray, matrix: np.ndarray):
        row_indices = np.nonzero(
            (matrix == row).all(axis=1)
        )[0]

        # assumes that matrix contains only unique rows => either found 1 row or none
        if not row_indices.size:
            return None
        return row_indices[0]

    def _get_added_and_changed(self, batch):
        replay_keys = np.packbits(self._replay_batch)
        batch_keys = np.packbits(batch)

        added_indices = []              # new_batch-wise
        changed_reward_indices = []     # replay_buffer-wise

        for i in range(batch.x.shape[0]):
            sample_key = batch_keys[i]

            ind = self._find_row_index(sample_key, replay_keys)
            if ind is None:
                # new
                added_indices.append(i)
            else:
                # double
                # todo: move to reward resolver
                if batch.r[ind] == 0 and self._replay_batch.r[ind] == 1:
                    changed_reward_indices.append(ind)

        return added_indices, changed_reward_indices

    def _add_batch_to_replay_buffer_backup(self, batch):
        replay_size = self._replay_batch.x.shape[0]

        # concatenate replay + batch
        x = np.concatenate((self._replay_batch.x, batch.x), axis=0)
        y = np.concatenate((self._replay_batch.y, batch.y), axis=0)
        r = np.concatenate((self._replay_batch.r, batch.r), axis=0)

        # # remove duplicates
        # x_filtered, y_filtered, r_filtered, unique_idx = self._get_uniques(x, y, r, return_index=True)
        # self._replay_buffer = ExperienceBatch(x_filtered, y_filtered, r_filtered)
        #
        # # find r = 0 duplicates (they can only locate in batch)
        # batch_size = len(batch.x)
        # concat_size = len(x)
        #
        # duplicates_mask = np.ones(concat_size, dtype=bool)
        # duplicates_mask[unique_idx] = False
        # no_reward_mask = r == 0
        # reward_renew_indices = np.nonzero(duplicates_mask & no_reward_mask)[0]
        # assert (reward_renew_indices >= replay_size).all()
        # reward_renew_samples = x[reward_renew_indices]
        #
        # # renew rewards to zero
        # replay_renewed_indices = []
        # for sample in reward_renew_samples:
        #     indices = np.nonzero((self._replay.x == sample).all(axis=1))[0]
        #     assert len(indices) == 1
        #     idx = indices[0]
        #     if self._replay.r[idx] != 0:
        #         self._replay.r[idx] = 0
        #         replay_renewed_indices.append(idx)
        # print('new r=0 samples overwritten: {}'.format(reward_renew_indices.size))
        #
        # # find non-duplicate indices in new batch (batch-based indexing)
        # new_batch_mask = unique_idx >= replay_size
        # new_non_duplicate_indices = unique_idx[new_batch_mask] - replay_size
        #
        # # find indices that will index constraints_buff + new_batch_unique synchronously with replay
        # constraints_unique_idx = unique_idx.copy()
        # constraints_unique_idx[new_batch_mask] = replay_size + np.arange(len(new_non_duplicate_indices))
        #
        # for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
        #     attr_batch = (batch.x[new_non_duplicate_indices],
        #                   batch.y[new_non_duplicate_indices, attr_idx])
        #     self._attr_mip_models[attr_idx].add_to_constraints_buff(attr_batch, constraints_unique_idx)
        #
        # reward_batch = (batch.x[new_non_duplicate_indices],
        #                 batch.r[new_non_duplicate_indices])
        # self._reward_mip_model.add_to_constraints_buff(reward_batch, constraints_unique_idx,
        #                                                replay_renewed_indices=replay_renewed_indices)


class SchemaSet:
    def __init__(self, schema_size=Constants.SCHEMA_VEC_SIZE, max_schema_count=Constants.L):
        self._W = np.empty((schema_size, max_schema_count), dtype=bool)
        self._count = 1
        self._max_count = max_schema_count

    def is_full(self):
        return self._count >= self._max_count

    @property
    def W(self):
        return self._W[:, :self._count]

    def predict(self, x):
        assert x.dtype == bool
        return ~(~x @ W)

    def add_schema(self, schema):
        if self._count < self._max_count:
            self._W[:, self._count] = schema
            self._count += 1

    def remove_schemas(self, schema_indices):
        for i in schema_indices:
            # remove i-th schema by replacing it w/ the last schema
            self._W[:, i] = self._W[:, self._count - 1]
            self._count -= 1


class AbstractMipModel(ABC):
    @property
    @abstractmethod
    def w(self):
        ...

    @abstractmethod
    def set_n_threads(self, n_threads=4):
        ...

    @abstractmethod
    def set_time_limit(self, time_limit_sec=2):
        ...

    @abstractmethod
    def add_constraint(self, constraint):
        ...

    @abstractmethod
    def set_min_sum_objective(self, summands):
        ...

    @abstractmethod
    def solve(self):
        ...

    def predict(self, x):
        w = self.w
        return (1 - x) @ w

    @staticmethod
    def get_one_target_constraint(pr):
        return -.1 <= pr <= .1

    @staticmethod
    def get_zero_target_constraint(pr):
        return pr >= .9

    @property
    def _arr_type(self):
        return np.int


class PythonMipModel(AbstractMipModel):
    def __init__(self, n_vars, emphasis=1):
        self._model = mip.Model(mip.MINIMIZE, solver_name=mip.CBC)
        self._model.verbose = False
        self._model.emphasis = emphasis

        self._w = np.array([self._model.add_var(var_type='B') for _ in range(n_vars)])

    @property
    def w(self):
        return self._w

    def set_n_threads(self, n_threads=4):
        self._model.threads = n_threads

    def set_time_limit(self, time_limit_sec=2):
        self._time_limit_sec = time_limit_sec

    def add_constraint(self, constraint):
        self._model.add_constr(constraint)

    def remove_constraints(self, constraints):
        self._model.remove(constraints)

    def set_min_sum_objective(self, summands):
        self._model.objective = mip.minimize(mip.xsum(summands))

    def solve(self):
        status = self._model.optimize(max_seconds=self._time_limit_sec)
        # todo: move print debug info to the higher levels
        self._print_solution_info(status)
        if not self._is_feasible(status):
            return None

        schema = self._get_solution()
        loss = self._get_loss()
        return schema, loss

    def _get_solution(self):
        return np.array([v.x for v in self._model.vars], dtype=self._arr_type)

    def _get_loss(self):
        return self._model.objective_value

    def _get_best_possible_loss(self):
        return self._model.objective_bound

    @classmethod
    def _is_feasible(cls, status):
        return status in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE]

    def _print_solution_info(self, status):
        if status == mip.OptimizationStatus.OPTIMAL:
            loss = self._get_loss()
            print_str = f'Optimal solution cost {loss} found'
        elif status == mip.OptimizationStatus.FEASIBLE:
            loss = self._get_loss()
            best_possible_loss = self._get_best_possible_loss()
            print_str = f'Sol.cost {loss} found, best possible: {best_possible_loss}'
        elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
            best_possible_loss = self._get_best_possible_loss()
            print_str = f'No feasible solution found, lower bound is: {best_possible_loss}'
        else:
            print_str = 'Optimization FAILED.'

        print(print_str)


class SchemaMipSolver:
    def __init__(self, n_vars=Constants.SCHEMA_VEC_SIZE):
        self._model = PythonMipModel(n_vars)
        self._one_target_constraints = []

    def add_zero_target_constraints(self, zero_target_x):
        zero_target_predictions = self._model.predict(zero_target_x)
        constraints = map(self._model.get_zero_target_constraint, zero_target_predictions)

        for constraint in constraints:
            self._model.add_constraint(constraint)

    def _add_one_target_constraints(self, one_target_x):
        one_target_predictions = self._model.predict(one_target_x)
        constraints = map(self._model.get_one_target_constraint, one_target_predictions)

        for constraint in constraints:
            self._model.add_constraint(constraint)
            self._one_target_constraints.append(constraint)

    def _remove_solved_constraints(self):
        self._model.remove_constraints(self._one_target_constraints)
        self._one_target_constraints.clear()

    @staticmethod
    def _get_initial_solved_set(one_targets):
        one_target_indices = np.nonzero(one_targets)[0]
        i = np.random.choice(one_target_indices)
        return [i]

    def _get_solved_by_schema(self, x, schema):
        prediction = (1 - x) @ schema
        return np.nonzero(prediction == 0)

    def _solve(self):
        result = self._model.solve()
        # if not feasible retry with extended time_limit

        return result

    def _find_cluster(self, x, y):
        one_target_mask = y
        solved_indices = self._get_initial_solved_set(one_target_mask)
        self._add_one_target_constraints(x[solved_indices])

        self._model.set_min_sum_objective(predicts[one_target_mask])
        res = self._solve()
        self._remove_solved_constraints()

        return res

    def _simiplify_schema(self, x, y, schema):
        solved_indices = self._get_solved_by_schema(x, schema)
        self._add_one_target_constraints(x[solved_indices])

        self._model.set_min_sum_objective(self._model.w)
        res = self._solve()
        assert res is not None, '!!! Cannot simplify !!! At least one solution should exist'

        self._remove_solved_constraints()
        return res, solved_indices

    def find_new_schema(self, x, y):
        result = self._find_cluster(x, y)
        if result is None:
            return None

        schema, loss = result
        result, solved_indices = self._simiplify_schema(x, y, schema)

        schema, loss = result
        return schema, solved_indices

    def _binarize_schema(self, schema):
        # todo: remove if not needed
        threshold = 0.5
        return schema > threshold


class SchemaSetLearner:
    def __init__(self, schema_size, max_schema_count):
        self._schema_set = SchemaSet(schema_size=schema_size, max_schema_count=max_schema_count)
        self._schema_learner = SchemaMipSolver(n_vars=schema_size)

    def sync_with_changes(self, x, y):
        self._remove_false_positive_schemas(x, y)
        self._add_anti_false_positives_contraints(x, y)

    def learn(self, x, y):
        # shape: samples
        prediction = self._schema_set.predict(x).any(axis=1)
        zero_prediction_mask = ~prediction

        while not self._schema_set.is_full():
            print('finding cluster...')
            print('augmented_entities: {}'.format(x.shape[0]))
            print('zp pos samples: {}'.format(zero_prediction_mask.sum()))

            new_schema, solved_indices = self._schema_learner.find_new_schema(
                x[zero_prediction_mask], y[zero_prediction_mask]
            )
            self._schema_set.add_schema(new_schema)

            # solved indices are indices, where prediction is 1 (or True)
            zero_prediction_mask[solved_indices] = False
            print('Also added to solved: {}'.format(len(solved_indices)))

    def _remove_false_positive_schemas(self, x, y):
        # shape: samples x schemas
        prediction = self._schema_set.predict(x)
        false_positives = np.logical_and(prediction, ~y)

        # shape: num_schemas
        fp_schemas_mask = false_positives.any(axis=0)
        fp_schemas_indices = np.nonzero(fp_schemas_mask)

        self._schema_set.remove_schemas(fp_schemas_indices)

    def _add_anti_false_positives_contraints(self, x, y):
        zero_target_mask = ~y
        self._schema_learner.add_zero_target_constraints(x[zero_target_mask])


class SchemaNetworkLearner:
    def __init__(self,
            n_attr_schema_sets=Constants.N_PREDICTABLE_ATTRIBUTES,
            schema_size=Constants.SCHEMA_VEC_SIZE,
            max_schema_count=Constants.L
    ):
        self._replay_buffer = ReplayBuffer(n_features=schema_size, n_targets=n_attr_schema_sets)
        self._attr_schema_set_learners = [
            SchemaSetLearner(schema_size, max_schema_count) for _ in range(n_attr_schema_sets)
        ]
        self._rew_schema_set_learner = SchemaSetLearner(schema_size, max_schema_count)

        self._curr_iter = None
        self._visualizer = Visualizer(None, None, None)

    @property
    def learned_W(self):
        return [schema_set_learner._schema_set.W for schema_set_learner in self._attr_schema_set_learners]

    @property
    def learned_R(self):
        return self._rew_schema_set_learner._schema_set.W

    def set_curr_iter(self, curr_iter):
        self._curr_iter = curr_iter
        self._visualizer.set_iter(curr_iter)

    def take_batch(self, batch):
        self._replay_buffer.append_to_new_batch(batch)

    def learn(self):
        self._sync_replay_buffer()
        self._add_schema_sets()

        self._visualizer.visualize_and_dump_schemas(self.learned_W, self.learned_R)
        self._visualizer.visualize_replay_buffer(self._replay_buffer.replay_batch)

    def _sync_replay_buffer(self):
        new_indices, changed_reward_indices = self._replay_buffer.sync_replay_buffer()
        replay_batch = self._replay_buffer.replay_batch
        x = replay_batch.x
        r = replay_batch.r

        # Attr schemas: changes are new samples
        changed_indices = new_indices
        for schema_set_learner, y in zip(self._attr_schema_set_learners, replay_batch.y):
            schema_set_learner.sync_with_changes(x[changed_indices], y[changed_indices])

        # Rew schemas: changes are new samples and samples with changed rewards
        changed_indices = new_indices + changed_reward_indices
        self._rew_schema_set_learner.sync_with_changes(x[changed_indices], r[changed_indices])

    def _add_schema_sets(self):
        replay_batch = self._replay_buffer.replay_batch

        for schema_set_learner, y in zip(self._attr_schema_set_learners, replay_batch.y):
            schema_set_learner.learn(replay_batch.x, y)

        self._rew_schema_set_learner.learn(replay_batch.x, replay_batch.r)
