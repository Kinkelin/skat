from numba import njit, prange

from intrinsic import ctz
from skat import *
from skat_text import get_bitmap_text, GAME_TYPE_NAMES, get_game_name


class BasicBiddingAI:
    """
    Bids according to a basic algorithmic metric
    """

    def __init__(self):
        self.hand_cards = 0
        self.table_position = 0
        self.risk_taking = 1
        self.bid = self.game_type = self.extra_tier = self.confidence = 0
        self.use_skat = False
        self.picked_up_skat = False

    def receive_hand_cards(self, hand_cards, table_position, behaviour=1):
        """
        Receive information for this bidding phase.

        Args:
            hand_cards (int): np.uint32 bitmap
            table_position (int): Forehand (0), middlehand (1) or rearhand (2)
            behaviour (float): Recommended behaviour modifier. Used in larger scale simulations to generate varied data with some AI agents.
        """
        self.hand_cards = hand_cards
        self.table_position = table_position
        self.risk_taking = behaviour
        self.use_skat = False
        self.picked_up_skat = False

        # Calculate initial bid
        self.bid, self.game_type, self.extra_tier, self.confidence, self.use_skat = calculate_bid(hand_cards, table_position,0, True, 0, self.risk_taking)
        # print("initial bid",self.bid, "game_type",self.game_type, "extra_tier",self.extra_tier, "confidence", self.confidence, "use_skat",self.use_skat)
        pass

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
        self.bid = bid
        decision = self.use_skat
        self.picked_up_skat = decision
        return decision

    def announce(self, hand_cards):
        """
        This implementation looks at all 66 hand options that result from putting any 2 cards back into the Skat and picks the one that (it believes) brings the most points.

        Decides the following:

        game_type: 0-3 (Colors), 4 (Grand), 5 (Null)

        extra_tier: 0 (Normal), 7 (Schneider announced), 8 (Schwarz announced), 9 (Ouvert)

        hand_cards: New hand cards without Skat

        Args:
            hand_cards: Hand cards including skat if picked up, np.uint32 bitmap
        Returns:
            tuple: (game_type, extra_tier, hand_cards)
        """

        # If Skat wasn't picked up simply play what we bid before
        if not self.picked_up_skat:
            return self.game_type, self.extra_tier, self.hand_cards

        return calculate_announcement_with_skat(hand_cards, self.table_position, self.risk_taking, self.bid)


@njit(parallel=False)
def calculate_announcement_with_skat(hand_cards_with_skat, table_position, risk_taking, bid):
    """
    Looks at all options for putting back 2 cards into the Skat, picks the best one, and announces the game for it

    Args:
        hand_cards_with_skat: np.uint32 bitmap with 12 bits on
        table_position (int): Forehand (0), middlehand (1) or rearhand (2)
        risk_taking: 1 is normal
        bid: The bid that was already raised, announcement has to match this (or we need to play into an overbid and reach Schneider+ in the trick game)

    Returns:
        tuple[int, int, int]:
        - game_type: Colors (0-3), Grand (4) or Null (5)
        - extra_tier: None (0), Schneider announced (1), Schwarz announced (2), Ouvert (3)
        - hand: New hand without skat (np.uint32 bitmap)
    """
    #
    options = generate_hands_without_skat(hand_cards_with_skat)

    points = np.empty(66, dtype=np.uint32)
    game_type = np.empty(66, dtype=np.uint32)
    extra_tier = np.empty(66, dtype=np.uint32)
    confidence = np.empty(66, dtype=np.uint32)

    for i in prange(66):
        points[i], game_type[i], extra_tier[i], confidence[i], _ = calculate_bid(options[i], table_position, bid, False, hand_cards_with_skat, risk_taking)

    best = 0
    for i in range(66):
        if (confidence[i] >= 1 and points[i] > points[best]) or (confidence[i] > confidence[best] and confidence[best] < 1):
            best = i

    return game_type[best], extra_tier[best], options[best]


# Config for bid calculation
COLOR_TRUMP_EVALUATION = np.array([-100, -50, -20, -5, -2, 1, 2, 4, 6, 10, 100], dtype=np.int32)

RISK_COLOR_HAND = 3.75
RISK_COLOR_SCHNEIDER = 5.0
RISK_COLOR_SCHWARZ = 6.5

RISK_GRAND_HAND = 10.5
RISK_GRAND_SCHNEIDER = 13.75


@njit
def calculate_extra_tier(risk_taking, viability, spitze_trumps, count_trumps, color_spitzen_sum, spitze):
    extra_tier = 0
    hand = risk_taking * viability >= RISK_COLOR_HAND
    if hand:
        extra_tier = EXTRA_TIER_HAND
        schneider = risk_taking * viability >= RISK_COLOR_SCHNEIDER
        if schneider:
            extra_tier = EXTRA_TIER_SCHNEIDER
            schwarz = viability >= RISK_COLOR_SCHWARZ
            if schwarz:
                extra_tier = EXTRA_TIER_SCHWARZ
                ouvert = spitze_trumps >= 5 and count_trumps >= 6 and (color_spitzen_sum - spitze) == 10 - count_trumps
                if ouvert:
                    extra_tier = EXTRA_TIER_OUVERT
    return extra_tier


@njit
def calculate_extra_tier_grand(count_jacks, jacks_dominance, risk_taking, viability, color_spitzen_sum):
    extra_tier = 0
    hand = risk_taking * viability >= RISK_GRAND_HAND
    if hand:
        extra_tier = EXTRA_TIER_HAND
        schneider = risk_taking * viability >= RISK_GRAND_SCHNEIDER
        if schneider:
            extra_tier = EXTRA_TIER_SCHNEIDER
            schwarz = jacks_dominance and color_spitzen_sum == 6
            if schwarz:
                extra_tier = EXTRA_TIER_SCHWARZ
                ouvert = count_jacks == 4 and color_spitzen_sum == 6
                if ouvert:
                    extra_tier = EXTRA_TIER_OUVERT

    return extra_tier


@njit
def calculate_overbid_tier(game_type, tier, extra_tier, minimum_bid):
    if minimum_bid <= 18:
        return 0
    overbid_tier = np.uint32(0)

    while (tier + extra_tier + overbid_tier) * BIDDING_BASE_VALUES[game_type] < minimum_bid:
        overbid_tier += 1

    verbose = False
    if verbose:
        print("Overbid calculation", GAME_TYPE_NAMES[game_type], "tier", tier, "extra_tier", extra_tier, "overbid_tier", overbid_tier, "points",
              (tier + extra_tier + overbid_tier) * BIDDING_BASE_VALUES[game_type], "minimum bid", minimum_bid)
    return overbid_tier

@njit
def calculate_null_overbid_tier(extra_tier, minimum_bid):
    points = BIDDING_NULL[extra_tier]
    if minimum_bid > points:
        return 2 # Null ouvert
    return 0

@njit
def get_null_overbid_punishment(extra_tier, overbid_tier):
    if overbid_tier == 0:
        return 0
    if extra_tier + overbid_tier <= EXTRA_TIER_NULL_HAND_OUVERT:
        return 25
    return 1000

@njit
def get_overbid_punishment(extra_tier, overbid_tier):
    if overbid_tier + extra_tier > EXTRA_TIER_OUVERT:
        return 10_000
    return 25 * overbid_tier

@njit
def calculate_null_color_gaps(hand_cards):
    null_color_gaps = np.zeros(4, dtype=np.float64)
    for i in range(4):
        opponent_cards_under = 0
        for j in range(8):
            card_id = get_card_id(i, RANKS_NULL_POSITION[j])
            card_present = is_card_present(hand_cards, card_id)
            if card_present:
                if opponent_cards_under > 0:
                    null_color_gaps[i] += 1
                    if opponent_cards_under > 1:
                        null_color_gaps[i] += 0.1
                opponent_cards_under = min(opponent_cards_under, 0) - 1 # A gap of one between cards is fine (7, 9, Bauer steht wie eine Mauer)
            else:
                opponent_cards_under += 1
    return null_color_gaps

@njit
def calculate_null_color_gaps_lut(hand_cards):
    null_color_gaps = np.zeros(4, dtype=np.float64)
    for i in range(4):
        opponent_cards_under = 0
        color_null_ordered = extract_color_null_ordered(hand_cards, i) # color_null_ordered is a bitmap of length 8 now, representing the cards of the current color, in Null relevant order
        if color_null_ordered != 0: # no cards of this color -> no gaps
            for j in range(8):
                card_present = is_card_present(color_null_ordered, j)
                if card_present:
                    if opponent_cards_under > 0:
                        null_color_gaps[i] += 1
                        if opponent_cards_under > 1:
                            null_color_gaps[i] += 0.1
                    opponent_cards_under = min(opponent_cards_under, 0) - 1 # A gap of one between cards is fine (7, 9, Bauer steht wie eine Mauer)
                else:
                    opponent_cards_under += 1
    return null_color_gaps


@njit
def calculate_null_color_gaps_ctz(hand_cards):
    null_color_gaps = np.zeros(4, dtype=np.float64)
    for i in range(4):
        bits = extract_color_null_ordered(hand_cards, i)
        opponent_cards_under = 0

        while bits != 0:
            tz = ctz(bits)
            opponent_cards_under += tz

            if opponent_cards_under > 0:
                null_color_gaps[i] += 1.0
                if opponent_cards_under > 1:
                    null_color_gaps[i] += 0.1

            bits = bits >> (tz + 1)
            opponent_cards_under = min(opponent_cards_under, 0) - 1
    return null_color_gaps

@njit
def calculate_bid(hand_cards, table_position=2, minimum_bid=0, skat_unknown=True, hand_with_skat=0, risk_taking=np.float32(1.0)):
    """
    Calculates a bid for the given hand of cards. Will pass a bad hand if allowed.

    Phase 1: Data processing
     - Prepares various datapoints

    Phase 2: Calculate viability
     - Viability is a scoring, different for each game type, that takes various metrics into account
     - Extra tiers are calls to make if we are especially confident in a game type (Hand, announce Schneider etc.)
     - Overbid tiers are necessary if standard play in a game type doesn't reach enough points to match the minimum bid. Generally undesirable

    Phase 3: Judge confidence and make a decision
     - Combine viability and extra+overbid tiers of each game type into a comparable confidence score. >= 1 means fully confident.
     - Of all the game types it is confident in, it will call the one that brings the most points.
     - If it is not confident in any game type, it will pass if possible.
     - If passing is not possible it will play what it's most confident in out of the remaining options

    To see detailed logs set verbose = True and comment out the njit annotation.

    Args:
        hand_cards (int): np.uint32 bitmap of all 32 cards
        table_position (int): Forehand (0), middlehand (1) or rearhand (2)
        minimum_bid (int): the minimum bid it will raise. Passing is only possible if this is 0
        skat_unknown (bool): If the skat is yet to be revealed, which unlocks extra tiers
        hand_with_skat (int): np.uint32 bitmap of length 32, contains the hand + the Skat cards if skat is known
        risk_taking (float): How risk-taking the bidding can be, 1 is normal
    Returns:
        (bid, game_type, extra_tier, confidence, use_skat
    """
    verbose = False

    #
    # Phase 1: Data processing
    #
    passing_possible = minimum_bid == 0
    very_verbose = verbose and passing_possible
    hand_with_skat |= hand_cards  # this will now hold all known cards "in our possesion"
    bitmap_aces = extract_aces(hand_cards)
    count_aces = count_cards(bitmap_aces)

    bitmap_jacks = extract_jacks(hand_cards)
    bitmap_known_jacks = extract_jacks(hand_with_skat)
    jacks_trump_spitze = get_spitze(bitmap_known_jacks, 4)
    jacks_spitze = get_spitze(bitmap_jacks, 4, True)
    count_jacks = count_cards(bitmap_jacks)

    color_bitmaps = np.empty(4, dtype=np.uint32)
    color_spitzen = np.empty(4, dtype=np.uint32)
    color_trump_spitzen = np.empty(4, dtype=np.uint32)
    color_counts = np.empty(4, dtype=np.uint32)
    color_trump_counts = np.empty(4, dtype=np.uint32)
    easy_points = np.zeros(4, dtype=np.int32)
    empty_colors = 0
    freebies = 2 * np.int32(skat_unknown)
    for i in range(4):
        color_bitmaps[i] = extract_color_without_jack(hand_cards, i)
        color_spitzen[i] = get_spitze(color_bitmaps[i], 7, True)
        color_trump_spitzen[i] = get_spitze(combine_jacks_and_color(bitmap_jacks, extract_color_without_jack(hand_with_skat, i)), 11)
        color_counts[i] = count_cards(color_bitmaps[i])
        color_trump_counts[i] = count_jacks + color_counts[i]
        if color_spitzen[i] == 1:
            easy_points[i] += 11
        elif color_spitzen[i] >= 2:
            easy_points[i] += 21
        elif color_counts[i] < 5:
            if is_card_present(hand_cards, get_card_id(i, R_10)):
                easy_points[i] -= 15
                if freebies > 0:
                    freebies -= 1
                    easy_points[i] += 12
            if is_card_present(hand_cards, get_card_id(i, R_K)):
                easy_points[i] -= 6
            if is_card_present(hand_cards, get_card_id(i, R_Q)):
                easy_points[i] -= 4
            if color_counts[i] == 0:
                empty_colors += 1

    easy_points_total = easy_points.sum()
    color_spitzen_sum = color_spitzen.sum()
    jacks_dominance = count_jacks == 4 or (count_jacks == 3 and jacks_trump_spitze >= 1) or (count_jacks == 2 and jacks_trump_spitze == 2)
    jacks_strength = count_jacks / 4 + jacks_trump_spitze / 4

    # if we don't have enough trumps, empty colors become a weakness
    grand_empty_colors = empty_colors * (jacks_strength - 0.35)
    grand_bad_colors = 4 - count_aces - empty_colors


    null_color_gaps = calculate_null_color_gaps(hand_cards)
    null_gaps = null_color_gaps.sum()

    position_factor = np.float32(0)
    null_position_factor = np.float32(0)
    if table_position == FOREHAND:
        position_factor = 0.1
        null_position_factor = -0.15
    elif table_position == MIDDLEHAND:
        position_factor = -0.05

    null_risk_taking = risk_taking + null_position_factor
    risk_taking += position_factor

    if very_verbose:
        print(f"color_spitzen {color_spitzen} color_spitzen_sum {color_spitzen_sum}, color_trump_counts {color_trump_counts} easy_points {easy_points} empty_colors {empty_colors}")
        print(f"jacks_strength {jacks_strength} count_jacks {count_jacks} jacks_spitze {jacks_spitze} jacks_trump_spitze {jacks_trump_spitze} grand_empty_colors {grand_empty_colors}")
        print(f"position_factor {position_factor} null_position_factor {null_position_factor} null_gaps {null_gaps} null_color_gaps {null_color_gaps}")

    #
    # Phase 2: Calculate viability
    #
    viability = np.empty(6, dtype=np.float32)
    tier = np.zeros(6, dtype=np.uint32)
    extra_tier = np.zeros(6, dtype=np.uint32)
    overbid_tier = np.zeros(6, dtype=np.uint32)

    # Viability and tier
    for i in range(4):
        viability[i] = jacks_spitze / 3 + COLOR_TRUMP_EVALUATION[color_trump_counts[i]] + 0.5 * (color_spitzen_sum - color_spitzen[i]) + (
                    easy_points_total - easy_points[i]) / 20 + empty_colors + count_aces / 2.0
        tier[i] = color_trump_spitzen[i] + 1

    viability[GRAND] = 2 * jacks_strength + count_jacks + 0.5 * color_spitzen_sum + easy_points_total / 15 + grand_empty_colors - grand_bad_colors + count_aces
    tier[GRAND] = jacks_trump_spitze + 1
    viability[NULL] = 1 - (null_gaps - 1) / 5  # Null viability just looks at gaps

    # Extra tier
    if skat_unknown:
        for i in range(4):
            extra_tier[i] = calculate_extra_tier(
                risk_taking,
                viability[i],
                color_trump_spitzen[i],
                color_trump_counts[i],
                color_spitzen_sum,
                color_spitzen[i]
            )
        extra_tier[GRAND] = calculate_extra_tier_grand(
            count_jacks,
            jacks_dominance,
            risk_taking,
            viability[GRAND],
            color_spitzen_sum
        )
        if null_risk_taking * viability[NULL] >= 1:
            extra_tier[NULL] = EXTRA_TIER_NULL_HAND
    if null_gaps == 0:
        extra_tier[NULL] = EXTRA_TIER_NULL_HAND_OUVERT if skat_unknown else EXTRA_TIER_NULL_OUVERT # Null ouvert can be played after skat was picked up

    # Overbid tier
    for i in range(5):
        overbid_tier[i] = calculate_overbid_tier(i, tier[i], extra_tier[i], minimum_bid)
    overbid_tier[NULL] = calculate_null_overbid_tier(extra_tier[NULL], minimum_bid)

    if very_verbose:
        print("viability", viability,"tier", tier, "extra_tier", extra_tier, "overbid_tier", overbid_tier)

    #
    # Phase 3: Judge confidence and make a decision
    #

    # For final calculations we take into account potential hand improvements from the Skat
    if skat_unknown:
        risky = np.float32(1.33)
        risk_taking *= risky
        null_risk_taking *= risky

    # Assemble options
    option_extra_tier = extra_tier + overbid_tier

    option_points = np.empty(6, dtype=np.uint32)
    for i in range(5):
        option_points[i] = BIDDING_BASE_VALUES[i] * (tier[i] + extra_tier[i] + overbid_tier[i])
    option_points[NULL] = BIDDING_NULL[extra_tier[NULL] + overbid_tier[NULL]]

    option_confidence = np.empty(6, dtype=np.float32)
    for i in range(4):
        option_points[i] = BIDDING_BASE_VALUES[i] * (tier[i] + extra_tier[i] + overbid_tier[i])
        option_confidence[i] = viability[i] if viability[i] < 0 else risk_taking * viability[i] / RISK_COLOR_HAND
        if extra_tier[i] > 0:
            option_confidence[i] += extra_tier[i]
        option_confidence[i] -= get_overbid_punishment(extra_tier[i], overbid_tier[i])
    option_confidence[GRAND] = max(min((risk_taking ** 2) * viability[GRAND] / RISK_GRAND_HAND, 1.0), risk_taking * viability[GRAND] / RISK_GRAND_HAND) + extra_tier[GRAND]
    option_confidence[GRAND] -= get_overbid_punishment(extra_tier[GRAND], overbid_tier[GRAND])
    option_confidence[NULL] = null_risk_taking * viability[NULL] + extra_tier[NULL] - get_null_overbid_punishment(extra_tier[NULL], overbid_tier[NULL])

    # Choose from options
    bid = np.uint32(0)
    game_type = np.uint32(0)
    extra_tier = np.uint32(0)
    confidence = np.float32(-100_000)
    for i in range(6):
        if (option_points[i] > bid and option_confidence[i] >= 1) or ((not passing_possible) and confidence < 1 and option_confidence[i] >= confidence):
            bid = option_points[i]
            game_type = i
            extra_tier = option_extra_tier[i]
            confidence = option_confidence[i]

    normal_game = extra_tier == 0
    null_game_with_skat = game_type == NULL and (extra_tier == 0 or extra_tier == NULL_OUVERT)
    use_skat = skat_unknown and (normal_game or null_game_with_skat)
    if verbose:
        print(
            f"Bid: {bid} Game: {get_game_name(game_type, extra_tier)} Use skat: {use_skat} confidence: {confidence} options {option_confidence} for {get_bitmap_text(hand_cards, game_type)} {hand_cards}")

    return bid, game_type, np.uint32(extra_tier), confidence, use_skat

