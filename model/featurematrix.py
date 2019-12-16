import numpy as np
from model.constants import Constants
import environment.schema_games.breakout.constants as constants


class FeatureMatrix(Constants):
    def __init__(self, env):
        """
        :param env: object of class environment.schema_games.breakout.games
        """
        matrix = np.zeros((self.SCREEN_HEIGHT * self.SCREEN_WIDTH, self.M - 1))
        ones = np.zeros((self.SCREEN_HEIGHT * self.SCREEN_WIDTH, 1)) + 1
        self.matrix = np.concatenate((matrix, ones), axis=1)
        self.planned_action = None

        for ball in env.balls:
            if ball.is_entity:
                for state, eid in env.parse_object_into_pixels(ball):
                    pos = list(state.keys())[0][1]
                    print('ball', pos)
                    ind = self.transform_pos_to_index(pos)
                    self.matrix[ind][self.BALL_IDX] = 1
                    self.matrix[ind][self.VOID_IDX] = 0

        paddle_width = constants.DEFAULT_PADDLE_SHAPE[0]
        paddle_height = constants.DEFAULT_PADDLE_SHAPE[1]
        if env.paddle.is_entity:
            for state, eid in env.parse_object_into_pixels(env.paddle):
                # TODO: add parts of paddle
                pos = list(state.keys())[0][1]
                print('paddle', pos)
                for i in range(1 - paddle_width // 2, paddle_width // 2 + 1):
                    for j in range(1 - paddle_height // 2, paddle_height // 2 + 1):
                        ind = self.transform_pos_to_index((pos[0]+j, pos[1]+i))
                        self.matrix[ind][self.PADDLE_IDX] = 1
                        self.matrix[ind][self.VOID_IDX] = 0

        for wall in env.walls:
            if wall.is_entity:
                for state, eid in env.parse_object_into_pixels(wall):
                    pos = list(state.keys())[0][1]
                    # do not add lower line
                    if pos[0] < self.SCREEN_HEIGHT:
                        ind = self.transform_pos_to_index(pos)
                        self.matrix[ind][self.WALL_IDX] = 1
                        self.matrix[ind][self.VOID_IDX] = 0

        brick_height = constants.DEFAULT_BRICK_SHAPE[1]
        brick_width = constants.DEFAULT_BRICK_SHAPE[0]

        for brick in env.bricks:
            if brick.is_entity:
                for state, eid in env.parse_object_into_pixels(brick):
                    pos = list(state.keys())[0][1]
                    for i in range(-brick_height // 2 + 1, brick_height // 2 + 1):
                        for j in range(-brick_width // 2, brick_width // 2):
                            ind = self.transform_pos_to_index((pos[0] + i, pos[1] + j))
                            self.matrix[ind][self.BRICK_IDX] = 1
                            self.matrix[ind][self.VOID_IDX] = 0

    def transform_pos_to_index(self, pos):
        return pos[0] * self.SCREEN_WIDTH + pos[1]

    def transform_index_to_pos(self, index):
        return index // self.SCREEN_WIDTH, index % self.SCREEN_WIDTH

    def get_attribute_matrix(self):
        return self.matrix.copy()

    def _get_neighbours_with_action(self, ind, action, matrix=None, add_all_actions=False):

        if matrix is None:
            matrix = self.matrix

        pos = self.transform_index_to_pos(ind)
        x = pos[0]
        y = pos[1]
        res = []

        if add_all_actions:
            action_vec = np.ones(self.ACTION_SPACE_DIM)
        else:
            action_vec = np.eye(self.ACTION_SPACE_DIM)[action]

        zeros = np.zeros(self.M)

        # central entity is first
        res.append(matrix[self.transform_pos_to_index([x, y])])

        for i in range(-self.NEIGHBORHOOD_RADIUS, self.NEIGHBORHOOD_RADIUS + 1):
            for j in range(-self.NEIGHBORHOOD_RADIUS, self.NEIGHBORHOOD_RADIUS + 1):
                if x + i < 0 or x + i >= self.SCREEN_HEIGHT or y + j < 0 or y + j >= self.SCREEN_WIDTH:
                    if not (i == 0 and j == 0):
                        res.append(zeros)
                elif not (i == 0 and j == 0):
                    res.append(matrix[self.transform_pos_to_index([x + i, y + j])])

        res.append(action_vec)

        return np.concatenate(res)

    def transform_matrix_with_action(self, action, custom_matrix=None, add_all_actions=False):
        if custom_matrix is not None:
            matrix = custom_matrix
        else:
            matrix = self.matrix

        return np.array([self._get_neighbours_with_action(i, action, matrix=matrix, add_all_actions=add_all_actions)
                         for i in range(0, self.SCREEN_HEIGHT * self.SCREEN_WIDTH)])
