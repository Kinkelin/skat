import numpy as np
from numba import njit

# Neural network input for card playing phase
# The model receives game state from its POV and also secondary information, that a human player can count/calculate from game state

FEATURES = np.empty((0, 3), dtype=np.int32) # Holds start [0] and end [1] indices for all features and length [2]
SIZE = 0

def _add_feature(length):
    global SIZE, FEATURES
    index = SIZE
    feature_id = FEATURES.shape[0]
    SIZE += length
    FEATURES = np.vstack([FEATURES, [index, index+length, length]])
    return feature_id

# A feature is a collection of datapoints in the observation space
# obs is a np.float32 array, but most features will only use boolean data
# For all features the observation space uses player indices as self 0, left 1 and right 2.

 # General game information
FEATURE_GAME_TYPE = _add_feature(6)  # 4 Colors, Grand, Null
FEATURE_EXTRA_TIER = _add_feature(4)  # Hand, Schneider angesagt, Schwarz angesagt, Ouvert --OR-- Null Hand, Null Ouvert, Null Hand Ouvert, -
FEATURE_SOLO_PLAYER = _add_feature(3)  # Which player is playing solo (Alleinspieler)

# Bidding results
FEATURE_BIDS = _add_feature(3)  # (float) Normalized point value of bidding for each player
FEATURE_BIDS_GAME_TYPE = _add_feature(3 * 6) # 4 Colors, Grand, Null, when ambiguity lower tier is assumed (so Grand T2 instead of Diamonds T4 for example)
FEATURE_BIDS_TIER = _add_feature(3 * 5) # Gewinnstufen (capped) (Not filled for Null, so that Jack understanding isn't affected)
FEATURE_BIDS_REAR_DECLINED = _add_feature(1) # Flag that shows if rear player only declined a higher bid, which gives its pass another meaning

# Current state
FEATURE_PLAYER_CARDS = _add_feature(3 * 32)  # Own hand and theoretically possible cards held by the other players
FEATURE_TRICKS = _add_feature(3 * 10 * 32)  # Index 0 is currently ongoing trick, index 1 is last completed and so on. Not yet played tricks just contain full zeros
FEATURE_TRICKS_PLAYED = _add_feature(1) # (float) Number of tricks already played (including ongoing trick). Normalized to 0-1
FEATURE_LEADER = _add_feature(3)  # Who plays/played the first card this trick
FEATURE_VALID_ACTIONS = _add_feature(32)  # which cards am I allowed to play

# Secondary information
FEATURE_COLORS_LEFT = _add_feature(3 * 5)  # Colors+Trumps potentially available per player
FEATURE_AUGEN = _add_feature(2) # (float) Augen collected per team, normalized to 0-1
FEATURE_SCHNEIDER_ESCAPED = _add_feature(1)  # Gegenspieler team already escaped Schneider
FEATURE_SCHWARZ_ESCAPED = _add_feature(1)  # Gegenspieler team already escaped Schwarz

@njit
def create_obs():
    return np.zeros(SIZE, np.float32)

@njit
def set_feature(obs, feature_id, data):
    """
    Fill data of a feature into the observation space

    Args:
        obs: observation space (np array)
        feature_id: FEATURE_ constant
        data: 1d numpy array
    """
    feature = FEATURES[feature_id]
    assert data.dtype == obs.dtype
    assert feature[2] == data.shape[0]
    obs[feature[0]:feature[1]] = data

@njit
def set_feature_bool(obs, feature_id, value):
    """
    Fills the feature of length 1, with 1 or 0, for true or false
    """
    assert FEATURES[feature_id, 2] == 1
    b = 1 if value else 0
    data = np.full(1, b, dtype=np.float32)
    set_feature(obs, feature_id, data)

@njit
def set_feature_scalar(obs, feature_id, scalar):
    """
    Fills the feature with new data, where the nth value (scalar) is set to 1 and everything else 0
    """
    assert scalar < FEATURES[feature_id, 2]
    data = create_empty_data(feature_id)
    data[scalar] = 1
    set_feature(obs, feature_id, data)

@njit
def set_feature_finish_trick(obs, trick):
    """
    Fills FEATURE_TRICKS with the latest (finished) trick cards and updates FEATURE_TRICKS_PLAYED.
    Called when a trick has been finished.
    Does not move older tricks.

    Args:
        obs:
        trick: array of length 3 with card ids. No card is represented by value < 0

    """
    tricks_played_index = FEATURES[FEATURE_TRICKS_PLAYED, 0]
    obs[tricks_played_index] += 0.1

    tricks_index = FEATURES[FEATURE_TRICKS, 0]
    for i in range(3):
        if trick[i] >= 0:
            obs[tricks_index + i * 32 + trick[i]] = 1

@njit
def set_feature_add_trick(obs, trick):
    """
    Moves existing tricks backwards in FEATURE_TRICKS and fills the new (unfinished) trick to position reserved for current trick.
    This will be called directly before trainee is expected to play a card.
    Args:
        obs:
        trick: array of length 3 with card ids. No card is represented by value < 0
    """
    feature = FEATURES[FEATURE_TRICKS]
    obs[feature[0] + 3 * 32:feature[1]] = obs[feature[0]:feature[0] + 3 * 9 * 32]
    obs[feature[0]:feature[0] + 3 * 32] = 0
    for i in range(3):
        if trick[i] >= 0:
            obs[feature[0] + i * 32 + trick[i]] = 1

@njit
def create_empty_data(feature_id):
    feature = FEATURES[feature_id]
    return np.zeros(feature[2], dtype=np.float32)
