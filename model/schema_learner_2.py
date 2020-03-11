from collections import namedtuple
from typing import List, Tuple

import mip.model as mip
import numpy as np

from model.constants import Constants
from model.visualizer import Visualizer

ExperienceBatch = namedtuple('ExperienceBatch', ['x', 'y', 'r'])


class ReplayBuffer:
    """
    Encapsulates collecting experience data into replay buffer, handling a) sample uniqueness constraint,
    b) r=0 samples priority over the same samples w/ r>0
    """

    _new_batch_buffer: List[ExperienceBatch]        # temporal buffer for new data
    _replay_batch: ExperienceBatch                  # all unique experience data

    def __init__(self, n_features=Constants.SCHEMA_VEC_SIZE, n_targets=Constants.N_PREDICTABLE_ATTRIBUTES):
        self._new_batch_buffer = []
        self._replay_batch = ExperienceBatch(
            np.empty((0, n_features), dtype=bool),
            np.empty((0, n_targets), dtype=bool),
            np.empty((0), dtype=bool)
        )

        packed_bits_len = (n_features - 1) // 8 + 1
        self._replay_batch_keys = np.empty((0, packed_bits_len), dtype=np.uint8)

    @property
    def replay_batch(self) -> ExperienceBatch:
        return self._replay_batch

    def sort(self):
        self._replay_batch_keys, indices = np.unique(self._replay_batch_keys, axis=0, return_index=True)
        x, y, r = self._replay_batch.x, self._replay_batch.y, self._replay_batch.r
        self._replay_batch = ExperienceBatch(
            x[indices], y[indices], r[indices]
        )

    def append_to_new_batch(self, batch):
        """Adds new batch to temporal buffer (not to the replay buffer!)."""
        for part in batch:
            assert part.dtype == bool

        x, y, r = batch
        if not x.size:
            return

        x, y, r = self._get_uniques(x, y, r)
        batch_of_uniques = ExperienceBatch(x, y, r)
        self._new_batch_buffer.append(batch_of_uniques)

    def sync_replay_buffer(self) -> Tuple[List[int], List[int]]:
        """
        Synchronizes replay buffer with temporal buffer of new batches.
        Returns tuple of new sample indices and indices with changed reward.
        """
        old_replay_buffer_len = self._replay_batch.x.shape[0]

        batch = self._flush_new_batch()
        added_indices, changed_reward_indices = self._get_added_and_changed(batch)

        # make new replay buffer batch
        x = np.concatenate((self._replay_batch.x, batch.x[added_indices]), axis=0)
        y = np.concatenate((self._replay_batch.y, batch.y[added_indices]), axis=0)
        r = np.concatenate((self._replay_batch.r, batch.r[added_indices]), axis=0)
        self._replay_batch = ExperienceBatch(x, y, r)

        # make added indices relative to the new replay buffer
        new_indices = list(range(old_replay_buffer_len, old_replay_buffer_len + len(added_indices)))

        # update rewards
        self._replay_batch.r[changed_reward_indices] = 0
        print('new r=0 samples overwritten: {}'.format(len(changed_reward_indices)))

        # both indices now replay-wise
        return new_indices, changed_reward_indices

    @staticmethod
    def _get_uniques(x, y, r):
        """Returns unique samples, based on `x` uniqueness."""

        # packbits makes bitset from bool array => comparison becomes faster
        keys = np.packbits(x, axis=1)
        _, unique_indices = np.unique(keys, axis=0, return_index=True)

        return x[unique_indices], y[unique_indices], r[unique_indices]

    def _flush_new_batch(self) -> ExperienceBatch:
        """Clears temporal batch buffer and returns its data as a single batch."""

        # clear temporal new batch buffer
        buffer = self._new_batch_buffer
        self._new_batch_buffer = []

        # check it's not empty
        assert buffer

        # sort it to keep r = 0 entries after deduplication
        buffer = sorted(buffer, key=lambda batch: batch.r[0])

        x, y, r = zip(*buffer)
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        r = np.concatenate(r, axis=0)

        x, y, r = self._get_uniques(x, y, r)
        return ExperienceBatch(x, y, r)

    @staticmethod
    def _find_row_index(row: np.ndarray, matrix: np.ndarray):
        row_indices = _1d_mask_to_indices(
            (matrix == row).all(axis=1)
        )

        # assumes that matrix contains only unique rows => either found 1 row or none
        if not row_indices.size:
            return None
        return row_indices[0]

    def _get_added_and_changed(self, batch):
        """
        Gets a) indices of new samples in the `batch`,
            b) indices of replay buffer samples with changed reward (based on the data from the `batch`)
        """
        replay_keys = self._replay_batch_keys
        batch_keys = np.packbits(batch.x, axis=1)

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
                if batch.r[i] == 0 and self._replay_batch.r[ind] == 1:
                    changed_reward_indices.append(ind)

        # cache new replay buffer keys
        self._replay_batch_keys = np.concatenate((replay_keys, batch_keys[added_indices]), axis=0)
        return added_indices, changed_reward_indices


class SchemaSet:
    """Represents the set of schema vectors."""

    _W: np.ndarray       # set of _bool_ schema vectors
    _count: int         # current schema count
    _max_count: int     # max possible schema count

    def __init__(self, schema_size=Constants.SCHEMA_VEC_SIZE, max_schema_count=Constants.L):
        self._W = np.empty((schema_size, max_schema_count), dtype=bool)
        self._count = 0
        self._max_count = max_schema_count

    def is_full(self):
        return self._count >= self._max_count

    @property
    def W(self):
        return self._W[:, :self._count]

    def predict(self, x):
        return _predict(x, self.W)

    def add_schema(self, schema):
        self._W[:, self._count] = schema
        self._count += 1

    def remove_schemas(self, schema_indices):
        for i in schema_indices:
            # remove i-th schema by replacing it w/ the last schema
            self._W[:, i] = self._W[:, self._count - 1]
            self._count -= 1


class PythonMipModel:
    """
    Wraps MIP optimization library methods into library-independent interface.
    """
    _model: mip.Model       # model implementing MIP optimization methods
    _time_limit_sec: int    # optimization time limit in seconds
    _w: np.ndarray          # model variables to optimize

    def __init__(self, n_vars):
        self._model = mip.Model(mip.MINIMIZE, solver_name=mip.CBC)
        self._model.verbose = 0
        self._time_limit_sec = 1

        self._w = np.array([self._model.add_var(var_type='B') for _ in range(n_vars)])

    @property
    def w(self):
        return self._w

    def set_n_threads(self, n_threads=4):
        self._model.threads = n_threads

    def set_time_limit(self, time_limit_sec):
        self._time_limit_sec = time_limit_sec

    def add_constraint(self, constraint: mip.LinExpr) -> mip.Constr:
        return self._model.add_constr(constraint)

    def remove_constraints(self, constraints):
        self._model.remove(constraints)

    def predict(self, x):
        w = self._w
        # x.dtype is bool, but (1 - x).dtype is int - exactly what's needed
        return (1 - x) @ w

    def get_sum(self, terms):
        return mip.xsum(terms)

    def set_minimize_objective(self, objective):
        self._model.objective = mip.minimize(objective)

    def solve(self):
        status = self._model.optimize(max_seconds=self._time_limit_sec, max_solutions=1)
        return status

    def get_solution(self):
        schema = self._get_solution()
        loss = self._get_loss()
        return schema, loss

    @staticmethod
    def get_one_target_constraint(pr):
        return pr == 0

    @staticmethod
    def get_zero_target_constraint(pr):
        return pr >= 1

    def _get_solution(self):
        # vars proved to be always int, i.e. no floats
        return np.array([v.x for v in self._model.vars], dtype=np.bool)

    def _get_loss(self):
        return self._model.objective_value

    def _get_best_possible_loss(self):
        return self._model.objective_bound

    @classmethod
    def is_feasible(cls, status):
        return status in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE]

    @classmethod
    def is_failed(cls, status):
        return not cls.is_feasible(status) and not status == mip.OptimizationStatus.NO_SOLUTION_FOUND

    def print_solution_info(self, status):
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
    """
    New schema finder.
    Handles consistency and lifecycle of the MIP model and its constraints.
    """
    _model: PythonMipModel                         # MIP model optimizer
    _one_target_constraints: List[mip.Constr]      # list of current "solved" constraints

    def __init__(self, n_vars=Constants.SCHEMA_VEC_SIZE, n_threads=Constants.N_LEARNING_THREADS):
        self._model = PythonMipModel(n_vars)
        self._model.set_n_threads(n_threads)

        self._one_target_constraints = []
        self._zero_target_constraints = []

    def add_zero_target_constraints(self, zero_target_x):
        """Adds anti- false positive constraints, i.e. constraints "schema shouldn't predict any zero-target\""""
        zero_target_predictions = self._model.predict(zero_target_x)
        constraints = map(self._model.get_zero_target_constraint, zero_target_predictions)

        for constraint in constraints:
            self._zero_target_constraints.append(self._model.add_constraint(constraint))

    def find_new_schema(self, x, y):
        """Finds new solution (=schema) on unsolved (=zero-prediction) samples."""
        result = self._find_cluster(x, y)
        if result is None:
            return None

        schema, loss = result
        print(f'Cluster schema: {_1d_mask_to_indices(schema)}')

        result = self._simplify_schema(x, schema)
        schema, loss = result
        print(f'Result schema:  {_1d_mask_to_indices(schema)}')

        solved_indices = self._get_solved_by_schema(x, schema)
        return schema, solved_indices

    def _find_cluster(self, x, y):
        """Finds some feasible solution and returns tuple (solution, loss)."""
        one_target_mask = y

        unsolved_sample_count = np.count_nonzero(one_target_mask)
        print(f'zp pos samples: {unsolved_sample_count}')
        if not unsolved_sample_count:
            # nothing to solve
            return None

        solved_indices = self._get_initial_solved_set(one_target_mask)
        self._add_one_target_constraints(x[solved_indices])

        objective = self._get_objective(x[one_target_mask])
        self._model.set_minimize_objective(objective)

        res = self._solve_with_retries()
        self._remove_solved_constraints()

        return res

    def _simplify_schema(self, x, schema):
        """Finds an equivalent more sparse solution (=schema) and returns tuple (solution, loss)."""
        solved_indices = self._get_solved_by_schema(x, schema)
        self._add_one_target_constraints(x[solved_indices])

        self._model.set_minimize_objective(self._model.get_sum(self._model.w))
        res = self._solve_with_retries()
        assert res is not None, '!!! Cannot simplify !!! At least one solution should exist'

        self._remove_solved_constraints()
        return res

    def _add_one_target_constraints(self, one_target_x):
        """Adds solved constraints, i.e. constraints "solution should predict these samples"."""
        one_target_predictions = self._model.predict(one_target_x)
        constraints = map(self._model.get_one_target_constraint, one_target_predictions)

        for constraint in constraints:
            # mip.LinExpr
            constraint = self._model.add_constraint(constraint)
            self._one_target_constraints.append(constraint)

    def _remove_solved_constraints(self):
        """Removes solved constraints from the model constraints."""
        self._model.remove_constraints(self._one_target_constraints)
        self._one_target_constraints.clear()

    @staticmethod
    def _get_initial_solved_set(one_target_mask):
        """Gets chosen random one-target sample index, that will be chosen as initial "solved"."""
        one_target_indices = _1d_mask_to_indices(one_target_mask)
        i = np.random.choice(one_target_indices)
        print(f'INIT SOLVED: {i}')
        return [i]

    @staticmethod
    def _get_solved_by_schema(x, schema):
        """Gets all "solved" sample indices, i.e. one-target samples, that's predicted by the `schema`."""
        prediction = _predict(x, schema)
        solved_indices = _1d_mask_to_indices(prediction)
        return solved_indices

    def _get_objective(self, one_target_x):
        return (~one_target_x).sum(axis=0) @ self._model.w

    def _solve_with_retries(self):
        """Tries to solve current optimization problem with multiple retries, sequentially increasing time limit."""
        time_limits_sec = [5, 15, 45]

        for tl_sec in time_limits_sec:
            self._model.set_time_limit(tl_sec)
            status = self._model.solve()

            # todo: move print debug info to the higher levels
            self._model.print_solution_info(status)

            if self._model.is_feasible(status):
                return self._model.get_solution()

            if self._model.is_failed(status):
                break
        return None


class SchemaSetLearner:
    """Represents one particular learnable schema set and methods to learn it."""
    _schema_set: SchemaSet              # current learned schema set
    _schema_learner: SchemaMipSolver    # new schema finder
    _name: str                          # just a label to distinguish schema sets (entity type or reward)

    def __init__(self, schema_size, max_schema_count, name):
        self._schema_set = SchemaSet(schema_size=schema_size, max_schema_count=max_schema_count)
        self._schema_learner = SchemaMipSolver(n_vars=schema_size)
        self._name = name

    @property
    def W(self):
        return self._schema_set.W

    def sync_with_changes(self, x, y):
        self._remove_false_positive_schemas(x, y)
        self._add_anti_false_positive_constraints(x, y)

    def learn(self, x, y):
        # shape: samples
        prediction = self._schema_set.predict(x).any(axis=1)
        zero_prediction_mask = ~prediction

        print(f'{self._name} =>')
        print(f'augmented_entities: {x.shape[0]}')

        while not self._schema_set.is_full():
            result = self._schema_learner.find_new_schema(x[zero_prediction_mask], y[zero_prediction_mask])
            if result is None:
                break

            new_schema, solved_indices = result
            self._schema_set.add_schema(new_schema)

            # uncheck solved indices in zero prediction mask (they're relative to zero prediction indices!)
            zero_prediction_indices = _1d_mask_to_indices(zero_prediction_mask)
            zero_prediction_mask[zero_prediction_indices[solved_indices]] = False

            print(f'Added to solved: {len(solved_indices)}')

        print(f'<= _____')

    def _remove_false_positive_schemas(self, x, y):
        zero_targets_mask = ~y

        # shape: samples x schemas
        prediction = self._schema_set.predict(x[zero_targets_mask])

        # shape: num_schemas
        fp_schemas_mask = prediction.any(axis=0)
        fp_schemas_indices = _1d_mask_to_indices(fp_schemas_mask)

        if fp_schemas_indices.size:
            self._schema_set.remove_schemas(fp_schemas_indices)
            print(f'Deleted incorrect schemas: {len(fp_schemas_indices)} of {self._name}')

    def _add_anti_false_positive_constraints(self, x, y):
        zero_target_mask = ~y
        self._schema_learner.add_zero_target_constraints(x[zero_target_mask])


class SchemaNetworkLearner:
    """
    Represents the whole set of learnable schema sets (aka schema network) and logic to learn it.
    """

    _replay_buffer: ReplayBuffer                        # whole experience replay buffer
    _attr_schema_set_learners: List[SchemaSetLearner]   # list of schema set learners, one for every attribute
    _rew_schema_set_learner: SchemaSetLearner           # schema set learner for rewards

    def __init__(
            self,
            n_attr_schema_sets=Constants.N_PREDICTABLE_ATTRIBUTES,
            schema_size=Constants.SCHEMA_VEC_SIZE,
            max_schema_count=Constants.L
    ):
        self._replay_buffer = ReplayBuffer(n_features=schema_size, n_targets=n_attr_schema_sets)
        self._attr_schema_set_learners = [
            SchemaSetLearner(schema_size, max_schema_count, Constants.ENTITY_NAMES[i])
            for i in range(n_attr_schema_sets)
        ]
        self._rew_schema_set_learner = SchemaSetLearner(schema_size, max_schema_count, 'REWARDS')

        self._curr_iter = None
        self._visualizer = Visualizer(None, None, None)

    @property
    def learned_W(self):
        """Gets current learned attribute schema sets."""
        return [schema_set_learner.W for schema_set_learner in self._attr_schema_set_learners]

    @property
    def learned_R(self):
        """Gets current learned reward schema set."""
        return self._rew_schema_set_learner.W

    def set_curr_iter(self, curr_iter):
        self._curr_iter = curr_iter
        self._visualizer.set_iter(curr_iter)

    def take_batch(self, batch):
        """Takes new batch into experience replay buffer."""
        self._replay_buffer.append_to_new_batch(batch)

    def learn(self):
        """Finds attribute and reward schema sets regarding the whole current state of replay buffer."""
        self._sync_replay_buffer()
        self._learn_schema_sets()

        self._visualizer.visualize_and_dump_schemas(self.learned_W, self.learned_R)
        self._visualizer.visualize_replay_buffer(self._replay_buffer.replay_batch)

    def _sync_replay_buffer(self):
        """Synchronizes new experience data with replay buffer and all learned schema sets."""
        new_indices, changed_reward_indices = self._replay_buffer.sync_replay_buffer()
        replay_batch = self._replay_buffer.replay_batch
        x = replay_batch.x
        r = replay_batch.r

        # Attr schemas: changes are new samples
        n = len(self._attr_schema_set_learners)
        changed_indices = new_indices
        for i in range(n):
            schema_set_learner = self._attr_schema_set_learners[i]
            y = replay_batch.y[:, i]
            schema_set_learner.sync_with_changes(x[changed_indices], y[changed_indices])

        # Rew schemas: changes are new samples and samples with changed rewards
        changed_indices = new_indices + changed_reward_indices
        self._rew_schema_set_learner.sync_with_changes(x[changed_indices], r[changed_indices])

    def _learn_schema_sets(self):
        """Applies schema sets learning step."""

        # NB: add `self._replay_buffer.sort()` here to make replay_batch order the same as in the old learner
        replay_batch = self._replay_buffer.replay_batch

        n = len(self._attr_schema_set_learners)
        for i in range(n):
            schema_set_learner = self._attr_schema_set_learners[i]
            y = replay_batch.y[:, i]
            schema_set_learner.learn(replay_batch.x, y)

        self._rew_schema_set_learner.learn(replay_batch.x, replay_batch.r)


def _predict(x: np.ndarray, schema_set: np.ndarray):
    # both dtype bool
    assert x.dtype == np.bool
    assert schema_set.dtype == np.bool
    return ~(~x @ schema_set)


def _1d_mask_to_indices(mask: np.ndarray):
    assert mask.ndim == 1
    return np.nonzero(mask)[0]
