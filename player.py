from environment.schema_games.breakout.games import StandardBreakout
from model.featurematrix import FeatureMatrix
import numpy as np
from model.inference import SchemaNetwork
import time
from model.constants import Constants
import random
import environment.schema_games.breakout.constants as constants
# from testing.testing import HardcodedSchemaVectors
# from model.schemanet import SchemaNet
from model.visualizer import Visualizer
from model.experience_buffer import ExperienceBuffer


class Player(Constants):
    def __init__(self, model, reward_model, game_type=StandardBreakout):
        self.model = model
        self.reward_model = reward_model
        self.game_type = game_type
        self._memory = []
        self.rewards = []
        self.planner = SchemaNetwork()

        self.model.load()
        self.reward_model.load(is_reward='reward')
        return

    # transform data for learning:
    def _x_add_prev_time(self, action):
        X = np.vstack([matrix.transform_matrix_with_action(action=action) for matrix in self._memory[:-1]])
        X_no_actions = (X.T[:-self.ACTION_SPACE_DIM]).T
        actions = (X.T[-self.ACTION_SPACE_DIM:]).T
        X = np.concatenate((X_no_actions[:-self.N], X_no_actions[self.N:], actions[self.N:]), axis=1)
        return X

    def _y_add_prev_time(self):
        return np.vstack([matrix.matrix.T for matrix in self._memory[2:]])

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

    def _get_action_for_reward(self, env, randomness=True):
        pos_ball = 0
        pos_paddle = 0

        if randomness:
            r = random.randint(1, 10)
            if r < 4:
                return 0

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

    def _min_mem(self):
        if len(self._memory) > 3:
            self._memory = self._memory[-3:]

    def save(self):
        print(self.rewards)
        self.reward_model.save(is_reward='reward')
        self.model.save()

    def play(self, game_type=StandardBreakout,
             learning_freq=30,
             log=False, cheat=False):

        vis_counter = 0

        flag = 0
        visualizer = Visualizer(None, None, None)
        buffer = ExperienceBuffer()

        for i in range(self.EP_NUM):
            env = game_type(return_state_as_image=False)
            env.reset()

            j = 0
            action = 0
            state, reward, done, _ = env.step(action)
            actions = [1, 2]
            while not done:
                vis_counter += 1

                self._memory.append(FeatureMatrix(env))
                self._min_mem()

                visualizer.set_iter(vis_counter)
                visualizer.visualize_env_state(FeatureMatrix(env).matrix)

                # learn new schemas
                if j > 1:

                    if j % 10 == 0:
                        buffer.log_reward()

                    # transform data for learning
                    X = self._x_add_prev_time(action)
                    y = self._y_add_prev_time()

                    buffer.add_attr(X, y)
                    buffer.add_reward(X, reward)

                    X_tmp, r_tmp = buffer.transform_state_to_check(X, reward)
                    self.reward_model._remove_wrong_schemas(X_tmp, r_tmp)

                    # learn env state:

                    if j % learning_freq == learning_freq - 4:
                        actions = [0, 1, 2]

                    if j % learning_freq == learning_freq - 1 :
                        self.model.fit(*buffer.get_attr_data())
                        self.reward_model.fit(*buffer.get_reward_data())

                    # make a decision
                    rand = random.randint(1, 10)
                    if flag < 5 and rand < 8:
                        if len(actions) > 0:
                            action = actions.pop(0)
                        else:
                            action = self._get_action_for_reward(env)

                    elif j >= learning_freq:
                        W = [w == 1 for w in self.model._W]
                        R = [self.reward_model._W[0] == 1, self.reward_model._W[1] == 1]

                        # W, R = HardcodedSchemaVectors.gen_schema_matrices()
                        if len(actions) > 0:
                            action = actions.pop(0)
                        elif all(w.shape[1] > 0 for w in W):
                            frame_stack = [obj.matrix for obj in self._memory[-self.FRAME_STACK_SIZE:]]
                            self.planner.set_weights(W, R)
                            self.planner.set_curr_iter(vis_counter)
                            actions = self.planner.plan_actions(frame_stack)
                            if actions is None:
                                actions = np.random.randint(low=0,
                                                            high=self.ACTION_SPACE_DIM,
                                                            size=1)
                            actions = list(actions)
                            action = actions.pop(0)
                    else:
                        action = self._get_action_for_reward(env)
                    self._memory.pop(0)


                j += 1
                # print('action:', action)
                state, reward, done, _ = env.step(action)
                if reward == 1:
                    actions = [0, 1, 2]
                    if flag == 3:
                        print('PLAYER CHANGED')
                    flag += 1

                elif reward == -1:
                    j = 0
                    actions = [0, 1, 2]
                    self._free_mem()

                self.rewards.append(reward)


        self.model.save()
