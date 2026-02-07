import numpy as np
from numba import njit

from skat import get_card_list, get_trump_cards_in_hand, RANKS_NULL_POSITION, is_card_present, get_card_id, must_follow_suit, NULL, get_card_color, get_trump_cards, GRAND, get_card_points, ACE, \
    get_card_rank, R_7, R_A, JACKS, JACK, TRUMP_CARDS, R_B
from skat_text import get_bitmap_text, get_card_name, get_list_text


class GreedyPlayingAI:
    """
    Always secure the trick with the highest possible card or add the most points if trick is secure.
    Play the lowest card if following suit in a Null game, otherwise throw the highest card possible
    """

    def __init__(self):
        self.game_type = 0
        self.extra_tier = 0
        self.solo_player = 0
        self.table_position = 0
        self.playing_solo = False
        self.verbose = False

    def start_playing(self, game_type, extra_tier, hand_cards, position, solo_player, ouvert_hand, bidding_history, behaviour=1):
        """
        Called at the start of the card playing phase to pass initial gamestate
        Args:
            game_type (int): Colors (0-3), Grand (4), Null (5)
            extra_tier (int): Normal - Ouvert (0-4) (See EXTRA_TIER_ constants for more details)
            hand_cards (int): bitmap of length 32
            position (int): 0-2 table position
            solo_player (int): 0-2 table position
            ouvert_hand (int): bitmap of length 32. Hand of the solo player, only given when game is Ouvert, otherwise 0
            bidding_history
            behaviour (float): 1 is normal
        """
        self.game_type = game_type
        self.extra_tier = extra_tier
        self.solo_player = solo_player
        self.table_position = position
        self.playing_solo = solo_player == position

    def play_card(self, hand_cards, valid_actions, current_trick, trick_giver, history):
        """
        Args:
            hand_cards (int): bitmap of length 32
            valid_actions (int): Hand cards that are legal to play (bitmap of length 32)
            current_trick: np array of length 3. Indices equal table positions. Not yet played cards are -1
            trick_giver: player who plays the first card this trick (0-2, table position)
            history:

        Returns:
            int: Card to play (Has to be one of valid actions)
        """
        #print("GreedyPlayingAI hand_cards", get_bitmap_text(hand_cards), "valid_actions", get_bitmap_text((valid_actions)))
        actions = get_card_list(valid_actions)
        trump_cards_available = get_card_list(get_trump_cards_in_hand(self.game_type, valid_actions))
        has_trump = len(trump_cards_available) > 0
        trick_position = (3 + self.table_position - trick_giver) % 3  # our position in the trick
        i_am_giver = trick_position == 0
        first_card = current_trick[trick_giver]
        follow_suit = False if i_am_giver else must_follow_suit(self.game_type, hand_cards, first_card)

        if self.game_type == NULL:
            return play_card_null_game(valid_actions, i_am_giver, follow_suit)

        highest_points = get_highest_points_action(actions)
        lowest = get_lowest_action(actions)
        highest_trump = get_highest_trump(trump_cards_available) if has_trump else None

        if self.verbose:
            print("GreedyPlayingAI highest_points",get_card_name(highest_points), "lowest", get_card_name(lowest), "highest_trump", get_card_name(highest_trump))

        if i_am_giver:
            if self.playing_solo and has_trump:
                return highest_trump
            else:
                return highest_points

        first_card_is_trump = is_card_present(get_trump_cards(self.game_type), first_card)

        if trick_position == 1:
            # Second on table

            if first_card_is_trump:
                if has_trump and get_highest_trump(np.array([first_card, highest_trump])) == highest_trump:
                    return highest_trump # Secure trick
                return lowest # Can't get this trick

            # First card is not trump
            if follow_suit:
                if highest_points > first_card:
                    return highest_points # Secure trick
                return lowest # Can't get this trick
            else:
                if has_trump:
                    return highest_trump # Secure trick
                return lowest # Can't get this trick

        # Third on table
        second_card = current_trick[(trick_giver + 1) % 3]
        second_card_is_trump = is_card_present(get_trump_cards(self.game_type), second_card)

        if first_card_is_trump or second_card_is_trump:
            if not has_trump:
                return lowest

            compare_cards = [highest_trump]
            if first_card_is_trump:
                compare_cards.append(first_card)
            if second_card_is_trump:
                compare_cards.append(second_card)
            if get_highest_trump(np.array([first_card, second_card, highest_trump])) == highest_trump:
                return highest_trump
            return lowest

        # No trumps on the table yet
        if has_trump:
            return highest_trump

        if follow_suit and highest_points > first_card and highest_points > second_card:
            return highest_points

        return lowest


@njit
def get_highest_points_action(actions):
    """

    Args:
        actions: np.array of card ids

    Returns:

    """
    best_card = -1
    best_points = -1
    for i in range(actions.shape[0] - 1, -1, -1):
        action = actions[i]
        points = get_card_points(action)
        if points > best_points:
            best_card = action
            best_points = points
            if points == 11:
                #print("get_highest_points_action ace", get_list_text(actions), get_card_name(best_card))
                return best_card
    #print("get_highest_points_action ", get_list_text(actions), get_card_name(best_card))
    return best_card

@njit
def get_lowest_action(actions):
    lowest_card = -1
    lowest_rank = R_B + 1
    for action in actions:
        rank = get_card_rank(action)
        if rank < lowest_rank:
            lowest_card = action
            lowest_rank = rank
            if rank == R_7:
                return action
    return lowest_card

@njit
def get_highest_trump(trump_cards):
    """
    Args:
        trump_cards: list of card ids

    Returns: card id
    """
    for jack in JACKS:
        if jack in trump_cards:
            return jack
    trump_cards.sort()
    return trump_cards[-1]

@njit
def play_card_null_game(actions, i_am_giver, follow_suit):
    if i_am_giver or follow_suit:
        return get_lowest_null_card(actions)
    return get_highest_null_card(actions)

@njit
def get_lowest_null_card(actions):
    for rank in RANKS_NULL_POSITION:
            for color in range(4):
                card_id = get_card_id(color, rank)
                if is_card_present(actions, card_id):
                    return card_id
    return 32 # Empty hand

@njit
def get_highest_null_card(actions):
    for r in range(RANKS_NULL_POSITION.shape[0]-1, -1, -1):
        rank = RANKS_NULL_POSITION[r]
        for color in range(4):
            card_id = get_card_id(color, rank)
            if is_card_present(actions, card_id):
                return card_id
    return 32 # Empty hand

@njit
def get_lowest_null_card_for_color(hand_cards, color):
    for rank in RANKS_NULL_POSITION:
        card_id = get_card_id(color, rank)
        if is_card_present(hand_cards, card_id):
            return card_id
    return 32 # Empty hand