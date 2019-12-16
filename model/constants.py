from environment.schema_games.breakout.constants import \
    BRICK_SIZE, ENV_SIZE, DEFAULT_HEIGHT, DEFAULT_WIDTH


class Constants:
    """
    N: number of entities
    M: number of attributes of each entity
    A: number of available actions
    L: number of schemas
    T: size of look-ahead window
    """
    DEBUG = False

    USE_LEARNED_SCHEMAS = False

    VISUALIZE_STATE = True
    VISUALIZE_SCHEMAS = False
    VISUALIZE_INNER_STATE = True
    VISUALIZE_BACKTRACKING_SCHEMAS = True
    VISUALIZE_BACKTRACKING_INNER_STATE = True

    if not DEBUG:
        if ENV_SIZE == 'DEFAULT':
            T = 130  # min 112
            EMERGENCY_REPLANNING_PERIOD = 30
        elif ENV_SIZE == 'SMALL':
            T = 50  # min 50
            EMERGENCY_REPLANNING_PERIOD = 10

        SCREEN_HEIGHT = DEFAULT_HEIGHT
        SCREEN_WIDTH = DEFAULT_WIDTH
        N = SCREEN_WIDTH * SCREEN_HEIGHT
        M = 5
        ACTION_SPACE_DIM = 3
        REWARD_SPACE_DIM = 2

        NEIGHBORHOOD_RADIUS = 2
    else:
        SCREEN_WIDTH = 3
        SCREEN_HEIGHT = 3

        N = 9  # SCREEN_WIDTH * SCREEN_HEIGHT
        M = 2
        T = 16
        ACTION_SPACE_DIM = 3
        REWARD_SPACE_DIM = 2

        NEIGHBORHOOD_RADIUS = 1

    L = 1000
    FILTER_SIZE = 2 * NEIGHBORHOOD_RADIUS + 1
    NEIGHBORS_NUM = FILTER_SIZE ** 2 - 1

    FAKE_ENTITY_IDX = N
    EPSILON = 0

    FRAME_STACK_SIZE = 2
    SCHEMA_VEC_SIZE = FRAME_STACK_SIZE * (M * (NEIGHBORS_NUM + 1)) + ACTION_SPACE_DIM
    TIME_SIZE = FRAME_STACK_SIZE + T

    # indices of corresponding attributes in entities' vectors
    BALL_IDX = 0
    PADDLE_IDX = 1
    WALL_IDX = 2
    BRICK_IDX = 3
    if not DEBUG:
        VOID_IDX = 4
    else:
        VOID_IDX = 1

    # action indices
    ACTION_NOP = 0
    ACTION_MOVE_LEFT = 1
    ACTION_MOVE_RIGHT = 2

    ENTITY_NAMES = {
        BALL_IDX: 'BALL',
        PADDLE_IDX: 'PADDLE',
        WALL_IDX: 'WALL',
        BRICK_IDX: 'BRICK',
    }

    REWARD_NAMES = {
        0: 'POSITIVE',
        1: 'NEGATIVE',
    }

"""
env changed constants:
BOUNCE_STOCHASTICITY = 0.25
PADDLE_SPEED_DISTRIBUTION[-1] = 0.90
PADDLE_SPEED_DISTRIBUTION[-2] = 0.10
_MAX_SPEED = 2
DEFAULT_BRICK_SHAPE = np.array([8, 4])
DEFAULT_NUM_BRICKS_ROWS = 6
DEFAULT_NUM_BRICKS_COLS = 11
"""