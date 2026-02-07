class SkatPlayer:

    def get_name(self):
        return "SkatPlayer"

    # Everything below implements the methods from BiddingAgent and TrickingAgent
    # Usually this can be done by wrapping instances of dedicated agents for each phase

    def receive_hand_cards(self,  hand_cards, table_position, behaviour=1):
        """
        Receive information for this bidding phase.

        Args:
            hand_cards (int): np.uint32 bitmap
            table_position (int): Forehand (0), middlehand (1) or rearhand (2) (this is a position relative to the bidding forehand player)
            behaviour (float): Recommended behaviour modifier. Used in larger scale simulations to generate varied data with some AI agents.
        """
        pass

    def say(self, next_bid, history):
        """
        Sagen oder passen

        Returns:
            int: bid (18, 20, etc.) 0 means pass
        """
        return

    def hear(self, bid, history):
        """
        Hören: Ja oder passen

        Returns:
            bool: Yes (True) or pass (False)

        """
        return

    def pickup_skat(self, highest_bid, history):
        """
        Args:
            history: The complete bidding history
        Returns:
            bool: if Skat is picked up (True) or instead Hand is played (False)
        """
        return False

    def announce(self, hand_cards):
        """
        Decides the following:

        game_type: 0-3 (Colors), 4 (Grand), 5 (Null)

        extra_tier: 0 (Normal), 7 (Schneider announced), 8 (Schwarz announced), 9 (Ouvert)

        hand_cards: New hand cards without Skat

        Args:
            hand_cards: Hand cards including skat if picked up, np.uint32 bitmap
        Returns:
            tuple: (game_type, extra_tier, hand_cards)
        """
        return

    def start_playing(self, game_type, extra_tier, hand_cards, position, solo_player, ouvert_hand, bidding_history, behaviour=1):
        """
        Called at the start of the card playing phase to pass initial gamestate
        Args:
            game_type (int): Colors (0-3), Grand (4), Null (5)
            extra_tier (int): Normal - Ouvert (0-4) (See EXTRA_TIER_ constants for more details)
            hand_cards (int): bitmap of length 32
            position (int): 0-2 table position (This can be an absolute table position, so it might differ from the table position in the bidding phase)
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
        return