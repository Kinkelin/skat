class BiddingAgent:

    # Geben-Hören-Sagen-Weitersagen

    def receive_hand_cards(self, hand_cards, table_position, behaviour=1):
        """
        Receive information for this bidding phase.

        Args:
            hand_cards (int): np.uint32 bitmap
            table_position (int): Forehand (0), middlehand (1) or rearhand (2)
            behaviour (float): Recommended behaviour modifier. Used in larger scale simulations to generate varied data with some AI agents.
        """
        pass

    def say(self, next_bid, history):
        """
        Sagen oder passen

            Returns:
                int: bid (18, 20, etc.) 0 means pass
        """
        return 0

    def hear(self, bid, history):
        """
        Hören: Ja oder passen

        Returns:
            bool: Yes (True) or pass (False)

        """
        return False

    def pickup_skat(self, bid, history):
        """
        Args:
            bid: The bid that the player agreed on and now has to match with points to win the game
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
        return 0, 0, hand_cards
