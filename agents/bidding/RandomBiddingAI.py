from skat import *

class RandomBiddingAI:
    """
    Submits a random bid that mirrors real world distribution
    """

    def __init__(self, rng_seed=None):
        self.rng = np.random.default_rng(rng_seed)

        self.game_type = 0
        self.extra_tier = 0
        self.bid = 0

    def receive_hand_cards(self, hand_cards, table_position, behaviour=1):
        """
        Receive information for this bidding phase.

        Args:
            hand_cards (int): np.uint32 bitmap
            table_position (int): Forehand (0), middlehand (1) or rearhand (2)
            behaviour (float): Recommended behaviour modifier. Used in larger scale simulations to generate varied data with some AI agents.
        """

        # Create a random bid immediately and say/hear/announce based on the stored bid
        self.game_type = self.rng.choice(7, p=[0.04, 0.05, 0.06, 0.07, 0.14, 0.01, 0.63])
        if self.game_type == 6:
            # Pass
            self.extra_tier = 0
            self.bid = 0
        elif self.game_type == NULL:
            self.extra_tier = self.rng.choice(4, p=[0.65, 0.2, 0.10, 0.05])
            self.bid = BIDDING_NULL[self.extra_tier]
        else:
            self.extra_tier = self.rng.choice(5, p=[0.69419, 0.28000, 0.02400, 0.00180, 0.00001])
            self.bid = BIDDING_BASE_VALUES[self.game_type] * (self.extra_tier + calculate_game_tier(self.game_type, hand_cards))

    def say(self, next_bid, history):
        """
        Sagen oder passen
            Returns:
                int: bid (18, 20, etc.) 0 means pass
        """
        return next_bid if next_bid <= self.bid else 0

    def hear(self, bid, history):
        """
        Hören: Ja oder passen

        Returns:
            bool: Yes (True) or pass (False)

        """
        return bid <= self.bid

    def simplified_bidding(self, hand_cards):
        """
        A simplified bidding phase, where every player directly decides on the maximum bid.

        Returns:
            tuple: (
            final bid (0, 18, 20, etc.),
            game_type,
            extra_tier
            )
        """
        return self.bid, self.game_type, self.extra_tier

    def pickup_skat(self, bid, history):
        """
        Args:
            bid: The bid that the player agreed on and now has to match with points to win the game
            history: The complete bidding history
        Returns:
            bool: if Skat is picked up (True) or instead Hand is played (False)
        """
        return self.extra_tier == 0

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
        cards = get_card_list(hand_cards)
        if self.extra_tier == 0:
            skat = self.rng.choice(cards, 2, False)
            for card in skat:
                hand_cards = remove_card(hand_cards, card)
        return self.game_type, self.extra_tier, hand_cards

