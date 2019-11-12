from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
import numpy as np
from model.schemanet import SchemaNet
from model.inference import SchemaNetwork
import time
from model.constants import Constants
import random
import environment.schema_games.breakout.constants as constants


class Player(Constants):
    def __init__(self, model, reward_model, game_type=StandardBreakout):
        self.model = model
        self.reward_model = reward_model
        self.game_type = game_type
        self._memory = []
        self.rewards = []
        return

    def _transform_to_array(self, l, pos=0, neg=0, ):
        print(l)
        return (np.zeros((l, self.M)) + np.array([pos, neg] + [0] * (self.M - 2))).T

    def _uniqy(self, X):
        if len(X) == 0:
            return np.array([])
        return np.unique(X, axis=0)

    # transform data for learning:
    def _x_add_prev_time(self, action):

        X = np.vstack((matrix.transform_matrix_with_action(action=action) for matrix in self._memory[:-1]))
        X_no_actions = (X.T[:-self.ACTION_SPACE_DIM]).T
        actions = (X.T[-self.ACTION_SPACE_DIM:]).T
        X = np.concatenate((X_no_actions[:-self.N], X_no_actions[self.N:], actions[self.N:]), axis=1)
        return X

    def _y_add_prev_time(self):
        return np.vstack((matrix.matrix.T for matrix in self._memory[2:]))

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

    # def _update_reward(self, ):

    def get_paddle_reward(self, env):

        pl, ph = constants.DEFAULT_PADDLE_SHAPE

        pos_ball = 0
        pos_paddle = 0
        for ball in env.balls:
            if ball.is_entity:
                for state, eid in env.parse_object_into_pixels(ball):
                    pos_ball = list(state.keys())[0][1]

        if env.paddle.is_entity:
            for state, eid in env.parse_object_into_pixels(env.paddle):
                pos_paddle = list(state.keys())[0][1]

        if pos_paddle[1] + pl // 2 >= pos_ball[1] >= pos_paddle[1] - pl // 2 and pos_ball[0] == pos_paddle[0] - 2:
            return 1
        return 0


    def _get_action_for_reward(self, env):
        pos_ball = 0
        pos_paddle = 0
        for ball in env.balls:
            if ball.is_entity:
                for state, eid in env.parse_object_into_pixels(ball):
                    pos_ball = list(state.keys())[0][1]

        if env.paddle.is_entity:
            for state, eid in env.parse_object_into_pixels(env.paddle):
                pos_paddle = list(state.keys())[0][1]

        if pos_ball[1] < pos_paddle[1]:
            return 1
        return 2

    def _free_mem(self):
        self._memory = []

    def play(self, game_type=StandardBreakout,
             learning_freq=3,
             log=False, cheat=False):
        old_state = []
        vis_counter = 0

        flag = 0

        # don't hardcode size!!
        length_a = 253
        length_e = 5
        X_global = np.zeros((1, length_a))
        X_reward = np.zeros((1, length_a))
        y_global = np.zeros((length_e, 1))
        y_reward = np.zeros((length_e, 1))

        for i in range(self.EP_NUM):
            env = game_type(return_state_as_image=False)
            # done = False
            env.reset()
            if cheat:
                self.model.add_cheat_schema()

            j = 0
            action = 0
            state, reward, done, _ = env.step(action)
            actions = []


            while not done:
                vis_counter += 1

                self._memory.append(FeatureMatrix(env))

                # learn new schemas
                if j > 1:

                    # transform data for learning
                    X = self._x_add_prev_time(action)
                    y = self._y_add_prev_time()

                    X_tmp, ind = np.unique(X, axis=0, return_index=True, )

                    X_global = np.concatenate((X_global, X[ind]), axis=0)
                    y_global = np.concatenate((y_global, y.T[ind].T), axis=1)

                    l, X_reward = self._check_for_update(X[ind], X_reward)
                    if l > 0:
                        y_r = self._transform_to_array(l, reward > 0, reward < 0)
                        y_reward = np.concatenate((y_reward, y_r), axis=1)

                    X_tmp, ind = np.unique(X_global, axis=0, return_index=True, )
                    X_global = X_global[ind]
                    y_global = (y_global.T[ind]).T

                    print('&&&&&&&&&&&&&&&', y_reward.shape, X_reward.shape, reward)

                    X_tmp, ind = np.unique(X_global, axis=0, return_index=True, )
                    X_global = X_global[ind]
                    y_global = (y_global.T[ind]).T
                    # learn env state:

                    print('fitted ok:', self.model.fit(X_global, y_global))
                    print('fitted reward ok:', self.reward_model.fit(X_reward, y_reward))

                    # make a decision
                    rand = random.randint(1, 10)
                    if flag == 0 and rand < 8:
                        action = self._get_action_for_reward(env)
                    else:
                        start = time.time()

                        W = [w == 1 for w in self.model._W]
                        R = [self.reward_model._W[0] == 1, self.reward_model._W[1] == 1]

                        if len(actions) > 0:
                            action = actions.pop(0)
                        elif all(w.shape[1] > 1 for w in W):
                            frame_stack = [obj.matrix for obj in self._memory[-self.FRAME_STACK_SIZE:]]
                            decision_model = SchemaNetwork(W, R, frame_stack)
                            decision_model.set_curr_iter(vis_counter)
                            actions = list(decision_model.plan_actions())
                            print('got actions', len(actions))
                            action = actions.pop(0)
                    self._memory.pop(0)


                j += 1
                print('action:', action)
                state, reward, done, _ = env.step(action)
                if reward == 1:
                    actions = []
                    if flag == 0:
                        print('PLAYER CHANGED')
                    flag = 1

                    #
                elif reward == -1:
                    j = 0
                    actions = []
                    self._free_mem()
                elif j > 200:
                    reward = self.get_paddle_reward(env)

                self.rewards.append(reward)

            if log:
                print('step:', i)

        self.model.save()
