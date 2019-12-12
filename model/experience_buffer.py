import numpy as np
from model.constants import Constants


class ExperienceBuffer(Constants):

    def __init__(self):
        length_a = self.M * (
                self.NEIGHBORHOOD_RADIUS * 2 + 1) ** 2 * self.FRAME_STACK_SIZE + self.ACTION_SPACE_DIM  # 253
        length_e = 5
        self.prev_states = np.zeros((1, length_a))
        self.prev_states_reward = np.zeros((1, length_a))

        self.next_state = np.zeros((length_e, 1))
        self.next_state_reward = np.zeros((length_e, 1))

    def get_attr_data(self):
        return self.prev_states, self.next_state

    def get_reward_data(self):
        return self.prev_states_reward, self.next_state_reward

    def add_attr(self, x, y):
        X_tmp, ind = np.unique(x, axis=0, return_index=True, )

        self.prev_states = np.concatenate((self.prev_states, x[ind]), axis=0)
        self.next_state = np.concatenate((self.next_state, y.T[ind].T), axis=1)

        X_tmp, ind = np.unique(self.prev_states, axis=0, return_index=True)
        self.prev_states = self.prev_states[ind]
        self.next_state = (self.next_state.T[ind]).T

    def _uniqy(self, X):
        if len(X) == 0:
            return np.array([])
        return np.unique(X, axis=0)

    def _transform_to_array(self, l, pos=0, neg=0, ):
        print(l)
        return (np.zeros((l, self.M)) + np.array([pos, neg] + [0] * (self.M - 2))).T

    def _check_for_update(self, X, old_state):
        old_state = np.array(old_state)
        update = []
        for entity in X:
            if not any((old_state == entity).all(1)):
                update.append(entity)
        update = self._uniqy(update)
        l = len(update)
        if l == 0:
            return l, old_state
        # print('update shape', l,  old_state.shape, np.array(update).shape)

        return l, np.concatenate((old_state, np.array(update)), axis=0)

    def add_reward(self, x, reward):
        X_tmp, ind = np.unique(x, axis=0, return_index=True, )
        l, self.prev_states_reward = self._check_for_update(x[ind], self.prev_states_reward)
        if l > 0:
            y_r = self._transform_to_array(l, reward > 0, reward < 0)
            self.next_state_reward  = np.concatenate((self.next_state_reward , y_r), axis=1)

