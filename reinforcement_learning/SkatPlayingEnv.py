import gymnasium as gym
import numpy as np
from numba import njit
import observation

from SkatGame import SkatGame
from agents.bidding.BasicBiddingAI import BasicBiddingAI
from agents.playing.GreedyPlayingAI import GreedyPlayingAI
from observation import *
from skat import deal_new_cards_as_bitmaps, deal_new_cards, BIDDING_VALUES, BIDDING_BASE_VALUES, BIDDING_NULL, NULL, GRAND, get_next_bid


class SkatPlayingEnv(gym.Env):
    """
    Train an AI for the playing phase of Skat.

    The environment uses randomly drawn cards and a simplified bidding phase using an Algorithmic bidding agent
    to set up varied games.

    In the playing phase (for now) two GreedyPlayingAI agents are used as opponents.
    Trained model is always index 0 at the table, the others are 1 and 2.

    This class implements similar logic to SkatGame, but in a structure useful for reinforcement learning.
    """
    def __init__(self):
        self.action_space = gym.spaces.Discrete(32)  # e.g., card index
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(observation.SIZE,), dtype=np.float32)
        self.obs = observation.create_obs()
        self.rng = None
        self.bidding_agent = BasicBiddingAI()
        self.playing_agent1 = GreedyPlayingAI()
        self.playing_agent2 = GreedyPlayingAI()
        self.cards = None
        self.forehand = 0

        self.solo_player = 0
        self.game_type = 0
        self.extra_tier = 0

        self.current_trick = np.full(3, -1, dtype=np.int64)
        self.current_trick_size = 3
        self.tricks = 0
        self.game_over = False

        #self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        print("Reset env")

        # Reset everything
        self.rng = self.np_random
        self.obs.fill(0)
        self.current_trick_size = 3 # Prompts the logic to start a new trick and thus the game
        self.tricks = 0
        self.game_over = False

        # Deal new cards
        self._deal_new_cards()

        # Simulate bidding
        self._bidding()

        # Start playing
        self._play_until_trainee()

        # Return initial observation for AI
        info = {}
        return self.obs.copy(), info

    def _play_until_trainee(self):
        if self.current_trick_size == 3:
            self.tricks += 1

            if self.tricks == 10:
                # All cards played, game regularly finished
                self.game_over = True
            else:

                # start new trick
                self.current_trick.fill(-1)
            pass
        else:
            # Finish trick. Trainee has already played.
            pass
        pass

    def step(self, action):

        # Apply trainee action
        self.game.play_card(player=1, card=action)

        # Simulate until trainee plays next card
        self._play_until_trainee()

        obs = self.obs.copy()
        reward = self._get_reward()
        terminated = self.game_over
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


    def render(self):
        pass

    def close(self):
        pass

    def _deal_new_cards(self):
        self.cards = deal_new_cards(self.rng)
        self.forehand = self.rng.integers(3)

    def _bidding(self):
        """
        Uses a simplified bidding process to determine game type and fill bids features in the observation space.
        Deals new cards as often as necessary to skip eingepasste games.
        """
        bids = np.empty(3, dtype=np.int64)
        game_types = np.empty(3, dtype=np.int64)
        extra_tiers = np.empty(3, dtype=np.int64)

        all_passed = True
        while all_passed:
            forehand = self.forehand
            middlehand = (forehand + 1) % 3
            rearhand = (forehand + 2) % 3

            for i in range(3):
                self.bidding_agent.receive_hand_cards(self.cards[i], (3+i-forehand) % 3)
                bids[i], game_types[i], extra_tiers[i] = self.bidding_agent.simplified_bidding(self.cards[i])

            all_passed = np.max(bids) == 0
            if all_passed:
                self._deal_new_cards()

        feature_bids = create_empty_data(FEATURE_BIDS)  # (float) Normalized point value of bidding for each player
        feature_bids_game_type = create_empty_data(FEATURE_BIDS_GAME_TYPE) # 4 Colors, Grand, Null
        feature_bids_tier = create_empty_data(FEATURE_BIDS_TIER) # Gewinnstufen (capped)

        rear_declined = False  # Flag that shows if rear player only declined a higher bid, which gives its pass another meaning

        public_bids = np.zeros(3, dtype=np.int64)
        if bids[middlehand] > bids[forehand]:
            public_bids[forehand] = bids[forehand]
            public_bid = get_next_bid(bids[forehand])
            if bids[rearhand] > bids[middlehand]:
                self.solo_player = rearhand  # Rear plays
                public_bids[middlehand] = bids[middlehand]
                public_bids[rearhand] = get_next_bid(bids[middlehand])
            else:
                self.solo_player = middlehand  # Middle plays
                public_bids[rearhand] = bids[rearhand] if bids[rearhand] > public_bid else 0
                public_bids[middlehand] = min(18, public_bid, bids[rearhand])
        else:
            public_bids[middlehand] = bids[middlehand]
            public_bid = bids[middlehand]
            if bids[rearhand] > bids[forehand]:
                self.solo_player = rearhand  # Rear plays
                public_bids[forehand] = bids[forehand]
                public_bids[rearhand] = get_next_bid(bids[forehand])
            else:
                self.solo_player = forehand # Fore plays
                public_bids[forehand] = min(18, public_bid, bids[rearhand])
                public_bids[rearhand] = bids[rearhand] if bids[rearhand] > public_bid else 0

        for i in range(3):
            bid = public_bids[i]
            if bid > 0:
                game_type, tier = _analyse_bid(bid)
                feature_bids_game_type[i * 6 + game_type] = 1
                feature_bids_tier[i * 5 + tier] = 1
                feature_bids[i] = _normalize_bid(bid)
            elif i == rearhand and bids[rearhand] > 0:
                rear_declined = True

        self.game_type = game_types[self.solo_player]
        self.extra_tier = extra_tiers[self.solo_player]

        set_feature(self.obs, FEATURE_BIDS, feature_bids)
        set_feature(self.obs, FEATURE_BIDS_GAME_TYPE, feature_bids_game_type)
        set_feature(self.obs, FEATURE_BIDS_TIER, feature_bids_tier)
        set_feature_bool(self.obs, FEATURE_BIDS_REAR_DECLINED, rear_declined)

        # General game information features
        set_feature_scalar(self.obs, FEATURE_GAME_TYPE, self.game_type)
        set_feature_scalar(self.obs, FEATURE_EXTRA_TIER, self.extra_tier)
        set_feature_scalar(self.obs, FEATURE_SOLO_PLAYER, self.solo_player)


@njit(inline='always')
def _normalize_bid(bid):
    return bid / BIDDING_VALUES[-1]

@njit
def _analyse_bid(bid):
    if bid in BIDDING_NULL:
        return NULL, -1
    game_type = -1
    for i in range(GRAND, -1, -1):
        if bid % BIDDING_BASE_VALUES[i] == 0:
            game_type = i
            break
    return game_type, bid / BIDDING_BASE_VALUES[game_type] - 2

# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/SkatPlaying-v0",
    entry_point=SkatPlayingEnv
)
