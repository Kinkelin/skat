from skat import get_card_list

class PlayingAgent:

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
        pass

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
        return get_card_list(valid_actions)[0]
