from collections import namedtuple
from itertools import chain
from abc import ABC, abstractmethod
import os
import numpy as np
import mip.model as mip
from model.constants import Constants
from model.visualizer import Visualizer


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

    def set_min_sum_objective(self, summands):
        self._model.objective = mip.minimize(mip.xsum(summands))

    def solve(self):
        status = self._model.optimize(max_seconds=self._time_limit_sec)
        if not self._is_feasible(status):
            return None

        schema = self._get_solution()
        loss = self._get_loss()
        return schema, loss

    def _get_solution(self):
        return np.array([v.x for v in self._model.vars], dtype=self._arr_type)

    def _get_loss(self):
        return self._model.objective_values[0]

    @classmethod
    def _is_feasible(cls, status):
        return status in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE]


class MipModel(Constants):
    """
    instantiated for single attr_idx
    """
    MAX_OPT_SECONDS = 5

    def __init__(self):
        self._model = mip.Model(mip.MINIMIZE, solver_name=mip.CBC)
        self._model.verbose = 0
        self._model.threads = self.N_LEARNING_THREADS

        self._w = [self._model.add_var(var_type='B') for _ in range(self.SCHEMA_VEC_SIZE)]
        self._constraints_buff = np.empty(0, dtype=object)

    def add_to_constraints_buff(self, batch, unique_idx, replay_renewed_indices=None):
        augmented_entities, target = batch
        batch_size = augmented_entities.shape[0]

        new_constraints = np.empty(batch_size, dtype=object)
        lin_combs = (1 - augmented_entities) @ self._w
        new_constraints[~target] = [lc >= 1 for lc in lin_combs[~target]]
        new_constraints[target] = [lc == 0 for lc in lin_combs[target]]

        concat_constraints = np.concatenate((self._constraints_buff, new_constraints), axis=0)
        self._constraints_buff = concat_constraints[unique_idx]

        if replay_renewed_indices is not None:
            for idx in replay_renewed_indices:
                self._constraints_buff[idx] = self._constraints_buff[idx] >= 1

    def optimize(self, objective_coefficients, zp_nl_mask, solved):
        model = self._model

        # add objective
        model.objective = mip.xsum(x_i * w_i for x_i, w_i in zip(objective_coefficients, self._w))

        # add constraints
        model.remove([constr for constr in model.constrs])

        constraints_mask = zp_nl_mask
        constraints_mask[solved] = True

        constraints_to_add = self._constraints_buff[constraints_mask]
        for constraint in constraints_to_add:
            model.add_constr(constraint)

        # optimize
        status = model.optimize(max_seconds=self.MAX_OPT_SECONDS)

        if status == mip.OptimizationStatus.OPTIMAL:
            print('Optimal solution cost {} found'.format(
                model.objective_value))
        elif status == mip.OptimizationStatus.FEASIBLE:
            print('Sol.cost {} found, best possible: {}'.format(
                model.objective_value, model.objective_bound))
        elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
            print('No feasible solution found, lower bound is: {}'.format(
                model.objective_bound))
        else:
            print('Optimization FAILED.')

        if status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
            schema_vec = np.array([v.x for v in model.vars])
        else:
            schema_vec = None
        return schema_vec


class SchemaSet:
    def __init__(self, schema_len=Constants.SCHEMA_VEC_SIZE, max_schema_count=Constants.L):
        self._W = np.ones((schema_len, max_schema_count), dtype=bool)
        self._count = 1
        self._max_count = max_schema_count

    @property
    def count(self):
        return self._count

    @property
    def W(self):
        return self._W[:, :self._count]

    def add_schema(self, schema):
        if self._count < self._max_count:
            self._W[:, self._count] = schema
            self._count += 1

    def remove_schemas(self, schema_indices):
        for i in schema_indices:
            # remove i-th schema by replacing it w/ the last schema
            self._W[:, i] = self._W[:, self._count - 1]
            self._count -= 1


ExperienceBatch = namedtuple('ExperienceBatch', ['x', 'y', 'r'])


class ReplayBuffer:
    def __init__(self, n_features=Constants.SCHEMA_VEC_SIZE, n_targets=Constants.N_PREDICTABLE_ATTRIBUTES):
        self._new_batch_buffer = []
        self._replay_buffer = ExperienceBatch(
            np.empty((0, n_features), dtype=bool),
            np.empty((0, n_targets), dtype=bool),
            np.empty((0), dtype=bool)
        )

    def _get_uniques(self, x, y, r, return_index=False):
        keys = np.packbits(x, axis=1)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)

        result = (x[unique_indices], y[unique_indices], r[unique_indices])
        if return_index:
            return result + (unique_indices, )
        return result

    def append_to_new_batch(self, batch):
        for part in batch:
            assert part.dtype == bool

        x, y, r = batch
        if x.size:
            return

        x, y, r = self._get_uniques(x, y, r)
        filtered_batch = ExperienceBatch(x, y, r)
        self._new_batch_buffer.append(filtered_batch)

    def flush_new_batch(self):
        if not self._new_batch_buffer:
            return None

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
        replay_keys = np.packbits(self._replay_buffer)
        batch_keys = np.packbits(batch)

        added_indices = []      # batch-wise
        changed_indices = []    # replay_buffer-wise

        for i in range(batch.x.shape[0]):
            sample_key = batch_keys[i]

            ind = self._find_row_index(sample_key, replay_keys)
            if ind is None:
                # new
                added_indices.append(i)
            else:
                # double
                # todo: move to reward resolver
                if sample_r[ind] == 0 and self._replay_buffer.r[ind] == 1:
                    changed_indices.append(ind)

        return added_indices, changed_indices

    def add_batch_to_replay_buffer(self, batch):
        added_indices, changed_indices = self._get_added_and_changed(batch)

        # add samples
        x = np.concatenate((self._replay_buffer.x, batch.x[added_indices]), axis=0)
        y = np.concatenate((self._replay_buffer.y, batch.y[added_indices]), axis=0)
        r = np.concatenate((self._replay_buffer.r, batch.r[added_indices]), axis=0)
        self._replay_buffer = ExperienceBatch(x, y, r)

        # change_indices
        self._replay_buffer.r[changed_indices] = 0


    def _add_batch_to_replay_buffer_backup(self, batch):
        replay_size = self._replay_buffer.x.shape[0]

        # concatenate replay + batch
        x = np.concatenate((self._replay_buffer.x, batch.x), axis=0)
        y = np.concatenate((self._replay_buffer.y, batch.y), axis=0)
        r = np.concatenate((self._replay_buffer.r, batch.r), axis=0)

        # remove duplicates
        x_filtered, y_filtered, r_filtered, unique_idx = self._get_uniques(x, y, r, return_index=True)
        self._replay_buffer = ExperienceBatch(x_filtered, y_filtered, r_filtered)

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

    @property
    def replay_buffer(self):
        return self._replay_buffer


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
