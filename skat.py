import numpy as np
import numba
from numba import njit
import intrinsic

# A card id is a number from 0-31, color * 8 + rank
# A group of cards (for example a hand) is represented as a np.uint32 bitmap, with a bit for every card id
NUMBER_OF_CARDS = 32

# Card ranks
R_7, R_8, R_9, R_Q, R_K, R_10, R_A, R_J = 0, 1, 2, 3, 4, 5, 6, 7

# Alternative names
JACK = BUBE = UNTER = R_U = R_B = R_J
QUEEN = OBER = R_O = R_Q
KING = KOENIG = R_K
ACE = ASS = DAUS = R_A

# Colors (also doubling as game type ids)
DIAMONDS = KARO = SCHELLEN = 0
HEARTS = HERZ = ROT = 1
SPADES = PIK = GRUEN = 2
CLUBS = KREUZ = EICHEL = 3

# Further game type flags
GRAND = 4
NULL = 5
HAND = NULL_HAND = 6
SCHNEIDER_ANNOUNCED = NULL_OUVERT = 7
SCHWARZ_ANNOUNCED = NULL_HAND_OUVERT = 8
OUVERT = 9

EXTRA_TIER_NONE = 0
EXTRA_TIER_HAND = EXTRA_TIER_NULL_HAND = 1
EXTRA_TIER_SCHNEIDER = EXTRA_TIER_NULL_OUVERT = 2
EXTRA_TIER_SCHWARZ = EXTRA_TIER_NULL_HAND_OUVERT = 3
EXTRA_TIER_OUVERT  = 4

NULL_RANK_ORDER = np.array([0, 1, 2, 5, 6, 3, 7, 4], dtype=np.uint8)
RANKS_NULL_POSITION = np.array([R_7, R_8, R_9, R_10, R_J, R_Q, R_K, R_A], dtype=np.uint32)

JACKS = BUBEN = UNTER = np.array([ 31 ,23, 15, 7], dtype=np.uint32)

# Point values
CARD_RANK_POINTS = np.array([0, 0, 0, 3, 4, 10, 11, 2], dtype=np.uint32)

# Table positions
FOREHAND = VORHAND = 0
MIDDLEHAND = MITTELHAND = 1
REARHAND = HINTERHAND = 2

# Player order
PLAYER_SELF = 0
PLAYER_LEFT = 1
PLAYER_RIGHT = 2

CARD_LOCATION_P0 = 0
CARD_LOCATION_P1 = 1
CARD_LOCATION_P2 = 2
CARD_LOCATION_SKAT = 3
CARD_LOCATION_SOLO = 4
CARD_LOCATION_TEAM = 5

RESULT_TEAM_WIN = 0
RESULT_SOLO_WIN = 1
RESULT_PASSED = 2

# Bidding
BIDDING_BASE_VALUES = np.array([9, 10, 11, 12, 24], dtype=np.uint32)
BIDDING_NULL = np.array([23, 35, 46, 59], dtype=np.uint32)
# GEWINNSTUFE is calculated as n + 2

def calculate_bidding_values():
    values = BIDDING_NULL.copy()
    for i in range(2, 19):
        for color in range(4):
            values = np.append(values, i * BIDDING_BASE_VALUES[color])
    for i in range(2, 12):
        values = np.append(values, i * BIDDING_BASE_VALUES[GRAND])
    return np.unique(values) # sort=True by default


BIDDING_VALUES = calculate_bidding_values()

@njit
def get_next_bid(bid):
    for i in range(BIDDING_VALUES.shape[0]):
        if BIDDING_VALUES[i] > bid:
            return BIDDING_VALUES[i]
    return 0




@njit(inline='always')
def get_card_color(card_id):
    """Get the color of a card.

    Args:
        card_id (int): Card ID (0-31).

    Returns:
        int: Card color (0-3).
    """
    return card_id >> 3 # shifting right by 3 bits, equivalent to floor division by 2^3, so equivalent to card_id // 8


@njit(inline='always')
def get_card_rank(card_id):
    """Get the rank of a card.

    Args:
        card_id (int): Card ID (0-31).

    Returns:
        int: Card rank (0-7).
    """
    return card_id & 7 # bitmask 7 in binary is 00111, & 7 isolates the three lowest bids, which equals modulo 8

@njit(inline='always')
def get_cards_that_have_been_removed(cards_a, cards_b):
    """
    This checks what cards have been present in a, but are no longer present in b
    Args:
        cards_a (int): bitmap of length 32
        cards_b (int): bitmap of length 32

    Returns:
        int: bitmap of length 32

    """
    return cards_a & (~cards_b & 0xFFFFFFFF)


@njit(inline='always')
def get_card_points(card_id):
    return CARD_RANK_POINTS[get_card_rank(card_id)]


@njit(inline='always')
def get_card_id(color, rank):
    # return color * 8 + rank

    # Equivalent with bit operators:
    return (color << 3) | rank

@njit(inline='always')
def is_card_present(cards, card_id):
    """Check if a card group contains a specific card.

    Args:
        cards (int): Bitmask representing a set of cards.
        card_id (int): Card position to check for shorter groups or Card ID (0-31) to check for a full bitmap of 32

    Returns:
        bool: True if the card is in the group, False otherwise.
    """
    return (cards & (1 << card_id)) != 0


@njit(inline='always')
def add_card(cards, card_id):
    """Add a card to a card group bitmap.

    Args:
        cards (np.uint32): Bitmap representing a set of cards.
        card_id (int): Card ID (0-31) to add.

    Returns:
        np.uint32: Updated card group bitmap.
    """
    return cards | (np.uint32(1) << card_id)


@njit(inline='always')
def remove_card(cards, card_id):
    """Remove a card from a card group bitmap.

    Args:
        cards (np.uint32): Bitmap representing a set of cards.
        card_id (int): Card ID (0-31) to remove.

    Returns:
        np.uint32: Updated card group bitmap.
    """
    return cards & ~(np.uint32(1) << card_id)


@njit
def get_bitmap(card_list):
    """
    Get the np.uint32 bitmap of length 32 for a list of card ids
    Args:
        card_list (int):
    Returns:
        int:
    """
    bitmap = np.uint32(0)
    for card_id in card_list:
        bitmap |= np.uint32(1) << card_id
    return bitmap


@njit
def get_card_list(bitmap):
    result = np.empty(32, dtype=np.uint32)
    count = 0
    for card_id in range(32):
        if bitmap & (1 << card_id):
            result[count] = card_id
            count += 1
    return result[:count]  # slice to actual length


@njit(inline='always')
def add_skat_to_hand(hand_cards, skat):
    """

    Args:
        hand_cards: bitmap of length 32
        skat: bitmap of length 32

    Returns:

    """
    return hand_cards | skat

@njit
def count_points(cards):
    """
    Args:
        cards: Bitmap of length 32

    Returns:

    """
    points = 0
    for i in range(32):
        card_points = get_card_points(i)
        if card_points > 0 and is_card_present(cards, i):
            points += card_points
    return points


@njit
def deal_new_cards_as_bitmaps():
    """
    Deal 10 cards to each player, keeping 2 in the Skat
    Returns:
         np.array([cards_p0, cards_p1, cards_p2, cards_skat]) - Bitmaps np.uint32 with length 32 each
    """
    deck = np.arange(NUMBER_OF_CARDS)
    np.random.shuffle(deck)
    cards_p0 = get_bitmap(deck[:10])
    cards_p1 = get_bitmap(deck[10:20])
    cards_p2 = get_bitmap(deck[20:30])
    cards_skat = get_bitmap(deck[30:])
    return np.array([cards_p0, cards_p1, cards_p2, cards_skat], dtype=np.uint32)

@njit
def deal_new_cards_from_deck(shuffled_deck):
    """
    Deal 10 cards to each player, keeping 2 in the Skat
    Args:
        shuffled_deck: array with numbers 0-31 in random order
    Returns:
         np.array([cards_p0, cards_p1, cards_p2, cards_skat]) - Bitmaps np.uint32 with length 32 each
    """
    deck = shuffled_deck
    cards_p0 = get_bitmap(deck[:10])
    cards_p1 = get_bitmap(deck[10:20])
    cards_p2 = get_bitmap(deck[20:30])
    cards_skat = get_bitmap(deck[30:])
    return cards_p0, cards_p1, cards_p2, cards_skat


def deal_new_cards(rng):
    """
       Deal 10 cards to each player, keeping 2 in the Skat
       Args:
           rng: Instance of np.random.Generator
       Returns:
            np.array([cards_p0, cards_p1, cards_p2, cards_skat]) - Bitmaps np.uint32 with length 32 each
       """
    deck = rng.arange(32)
    rng.shuffle(deck)
    return deal_new_cards_from_deck(deck)





@njit
def generate_hands_without_skat(hand_cards_with_skat):
    """
    Generate all 66 permutations of 10 card hands out of 12 cards

    Args:
        hand_cards_with_skat: np.uint32 bitmap of length 32

    Returns:
        np.array(66, dtype=np.uint32)

    """
    hand = hand_cards_with_skat
    # Get positions of bits that are on
    indices = numba.typed.List()
    for i in range(32):
        if (hand >> i) & 1:
            indices.append(i)

    hand_size = 12
    # assert len(indices) == hand_size

    # There will be 66 combinations
    out = np.empty(66, dtype=np.uint32)
    idx = 0
    for i in range(hand_size):
        for j in range(i + 1, hand_size):
            new_hand = hand & ~(1 << indices[i]) & ~(1 << indices[j])
            out[idx] = new_hand
            idx += 1
    return out


@njit
def random_cards(number_of_cards):
    """

    Args:
        number_of_cards (int): will result in that number of bits to be activated, representing the cards

    Returns:
        int: np.uint32 bitmap for length 32
    """
    n = 32  # total bits
    k = number_of_cards  # bits to set

    # Initialize array of all positions
    positions = np.arange(n, dtype=np.uint32)

    # Fisher-Yates style shuffle to pick k positions
    for i in range(k):
        # Pick a random index among remaining unchosen positions
        j = np.random.randint(i, n)
        # Swap positions[i] and positions[j]
        temp = positions[i]
        positions[i] = positions[j]
        positions[j] = temp

    # Now positions[0..k-1] are our chosen bit indices
    x = np.uint32(0)
    for i in range(k):
        x |= np.uint32(1) << positions[i]

    return x

@njit("uint32(uint32)")
def extract_jacks(cards):
    """

    Args:
        cards (int): np.uint32 bitmap for all 32 cards

    Returns:
        int: bitmap for the jacks only (4 bits)

    """
    return (
        ((cards >> 31) & 1) << 3 |
        ((cards >> 23) & 1) << 2 |
        ((cards >> 15) & 1) << 1 |
        ((cards >> 7)  & 1)
    )

@njit("uint32(uint32)")
def extract_aces(cards):
    """

    Args:
        cards (int): np.uint32 bitmap for all 32 cards

    Returns:
        int: bitmap for the aces only (4 bits)

    """
    return (
        ((cards >> 30) & 1) << 3 |
        ((cards >> 22) & 1) << 2 |
        ((cards >> 14) & 1) << 1 |
        ((cards >> 6)  & 1)
    )

@njit #("uint32(uint32, int)", inline='always')
def extract_color_without_jack(cards, color):
    """
    Args:
        cards (int): np.uint32 bitmap for all 32 cards

    Returns:
        int: bitmap for the cards of a color excluding the jack (7 bits)
    """
    return (cards >> (color << 3)) & 0x7F


@njit(inline='always')
def extract_color_with_jack(cards, color):
    """
    Args:
        cards (int): np.uint32 bitmap for all 32 cards

    Returns:
        int: bitmap for the cards of a color including the jack (8 bits)
    """
    return (cards >> (color << 3)) & 0xFF

@njit(inline='always')
def combine_jacks_and_color(jack_bits, color_bits):
    """
    Args:
        color_bits (int): bitmap for cards of a color without jack (7 bits)
        jack_bits (int): bitmap for the jacks only (4 bits)

    Returns:
        int: bitmap for all trumps in the color game_type (11 bits)

    """
    return color_bits | (jack_bits << 7)


@njit(inline='always')
def extract_color_trumps(cards, color):
    """

    Args:
        cards (int): np.uint32 bitmap for all 32 cards
        color (int): 0-4

    Returns:
        int: bitmap for all trumps in the color game_type (11 bits)

    """
    color_bits = extract_color_without_jack(cards, color)
    jack_bits = extract_jacks(cards)
    return color_bits | (jack_bits << 7)


# 2. The 32-bit Implementation
@njit
def get_spitze32(bitmap, bitmap_size, on_only=False):
    # Force input to 32-bit unsigned
    val = numba.uint32(bitmap)
    size = numba.uint32(bitmap_size)

    msb = (val >> (size - 1)) & 1

    if on_only:
        check_ones = True
    else:
        check_ones = (msb == 1)

    # 1. Align bits to the top of the 32-bit window
    # Note: explicit uint32 cast on the literal '32' helps Numba type inference
    shift_amount = 32 - size

    # 2. Shift and IMMEDIATELY cast back to uint32.
    # Without this, Numba promotes the result to int64, adding 32 extra leading zeros.
    shifted = numba.uint32(val << shift_amount)

    if check_ones:
        # 3. Invert and IMMEDIATELY cast back to uint32.
        # Python '~' results in a negative signed integer.
        # Casting to uint32 ensures we get the unsigned bitwise inverse (0xFFFFFFFF ^ val).
        shifted = numba.uint32(~shifted)

    # Now clz receives a pure uint32, so LLVM generates the 32-bit ctlz instruction
    cnt = intrinsic.clz(shifted)

    # cnt is now uint32. Clamp result.
    if cnt > size:
        return size

    return cnt


@njit
def get_spitze_neu(bitmap, bitmap_size, on_only=False):
    """
    Args:
        bitmap (int): limited bitmap for specific cards only
        bitmap_size (int): 4 or 11 for trump bitmaps for example

    Returns:
        int: Spitze, number of top trumps present/missing
    """
    # Cast to 64-bit to ensure consistent behavior for shifting and clz width
    # (Python ints are arbitrary width, but Numba usually treats them as int64)
    val = numba.int64(bitmap)
    size = numba.int64(bitmap_size)
    shift_amount = 64 - size
    shifted = val << shift_amount
    if on_only or (val >> (size - 1)) & 1:
        shifted = ~shifted
    cnt = intrinsic.clz(shifted)
    return min(size, cnt)


@njit
def get_spitze64(bitmap, bitmap_size, on_only=False):
    """
    Args:
        bitmap (int): limited bitmap for specific cards only
        bitmap_size (int): 4 or 11 for trump bitmaps for example

    Returns:
        int: Spitze, number of top trumps present/missing
    """
    # Cast to 64-bit to ensure consistent behavior for shifting and clz width
    # (Python ints are arbitrary width, but Numba usually treats them as int64)
    val = numba.int64(bitmap)
    size = numba.int64(bitmap_size)

    # Check the Most Significant Bit (MSB) relative to bitmap_size
    msb = (val >> (size - 1)) & 1

    # Determine if we are counting 1s (check_ones=True) or 0s (check_ones=False)
    if on_only:
        check_ones = True
    else:
        check_ones = (msb == 1)

    shift_amount = 64 - size
    shifted = val << shift_amount

    if check_ones:
        shifted = ~shifted

    # Count the leading zeros
    cnt = intrinsic.clz(shifted)

    # Clamp result to bitmap_size.
    # Example: If bitmap is 0 (size 4) and we count 0s, clz returns 64.
    # We only care about the first 'size' bits.
    if cnt > size:
        return size

    return cnt

@njit
def get_spitze(bitmap, bitmap_size, on_only=False):
    """
    Args:
        bitmap (int): limited bitmap for specific cards only
        bitmap_size (int): 4 or 11 for trump bitmaps for example

    Returns:
        int: Spitze, number of top trumps present/missing
    """

    spitzen_value = 1
    if not on_only:
        # Get MSB (Most significant bit -> Highest card)
        spitzen_value = (bitmap >> (bitmap_size - 1)) & 1

    cnt = 0
    for i in range(bitmap_size - 1, -1, -1):
        if ((bitmap >> i) & 1) == spitzen_value:
            cnt += 1
        else:
            break
    return cnt


@njit(inline='always')
def get_spitze_cgpt(bitmap, bitmap_size, on_only=False):
    bitmap = numba.int64(bitmap)
    shift = 64 - numba.int64(bitmap_size)
    bitmap <<= shift  # align logical MSB to bit 31
    # determine if we need to invert bits
    invert_mask = on_only | ((bitmap >> 63) & (~on_only))
    x = bitmap ^ (-invert_mask)  # XOR with all-ones if invert_mask=1
    return min(bitmap_size, intrinsic.clz(x))

@njit(inline='always')
def count_cards(card_bitmap):
    return intrinsic.popcount(card_bitmap)


#
# A completely overengineered lookup table solution to get the cards of a color Null ordered very quickly
#

# Full 8-bit LUT for a byte
def calculate_null_lookup_table():
    lut = np.zeros(256, dtype=np.uint8)
    for b in range(256):
        out = 0
        for i in range(8):
            if b & (1 << i):
                out |= (1 << NULL_RANK_ORDER[i])
        lut[b] = out
    return lut

NULL_LUT8 = calculate_null_lookup_table()

@njit(inline='always')
def extract_color_null_ordered(hand_cards, color):
    """
    Args:
        hand_cards (int): np.uint32 bitmap of length 32
        color (int): 0-3
    Returns:
        int: A bitmap of length 8, containing the null ordered cards of the color
    """
    color_mask = (hand_cards >> (color * 8)) & 0xFF
    return NULL_LUT8[color_mask]


def calculate_jacks_mask():
    mask = np.uint32(0)
    for jack in JACKS:
        mask = add_card(mask, jack)
    return mask

JACKS_MASK = calculate_jacks_mask()

def calculate_trump_cards():
    trump_cards = np.full(6, JACKS_MASK, dtype=np.uint32)
    for i in range(4):
        for rank in range(7):
            card_id = get_card_id(i, rank)
            trump_cards[i] = add_card(trump_cards[i], card_id)

    trump_cards[NULL] = 0 # No trumps in a Null game
    return trump_cards

TRUMP_CARDS = calculate_trump_cards()

@njit(inline='always')
def get_trump_cards(game_type):
    """Return bitmap of all trump cards for the given game type."""
    return TRUMP_CARDS[game_type]

@njit
def is_card_trump(game_type, card_id):
    return is_card_present(get_trump_cards(game_type), card_id)

@njit
def get_trump_cards_in_hand(game_type, hand_cards):
    """Return the cards in hand that are trump.
    Args:
        game_type (int): Color (0-3), Grand (4), Null (5)
        hand_cards (int): bitmap of length 32

    Returns: Hand cards that are trump (bitmap of length 32)
    """
    return hand_cards & get_trump_cards(game_type)


@njit
def calculate_card_groups():
    """
    Card groups are the cards that have to be played following another card of its group as the first card in a trick. Different groups for each game type.
    """
    groups = np.zeros((6, 5), dtype=np.uint32)

    # Fill color cards
    for i in range(6):
        ranks = 8 if i == NULL else 7
        for r in range(ranks):
            for c in range(4):
                groups[i, c] = add_card(groups[i, c], get_card_id(c, r))

    # Add Jacks to trump colors
    for i in range(4):
        groups[i, i] = groups[i, i] | JACKS_MASK

    # In Grand only Jacks are trump, so we need a fifth group. The fifth group stays empty for all other game types
    groups[GRAND, 4] = JACKS_MASK

    return groups


CARD_GROUPS = calculate_card_groups()


@njit
def get_valid_actions(game_type, first_card, hand_cards):
    g = 4 if game_type == GRAND and first_card in JACKS else get_card_color(first_card)
    follow_cards = CARD_GROUPS[game_type, g]
    following = follow_cards & hand_cards
    if following != 0:
        return following
    return hand_cards


@njit
def must_follow_suit(game_type, hand_cards, first_card):
    """

    Args:
        game_type (int): Color (0-3), Grand (4), Null (5)
        hand_cards (int): bitmap of length 32
        first_card (int): card id

    Returns:
        bool: If this hand of cards has to follow suit for the first card
    """
    g = 4 if game_type == GRAND and first_card in JACKS else get_card_color(first_card)
    follow_cards = CARD_GROUPS[game_type, g]
    return (follow_cards & hand_cards) != 0



@njit
def must_follow_suit_old(game_type, hand_cards, first_card) :
    color = get_card_color(first_card)
    rank = get_card_rank(first_card)
    if game_type == GRAND:
        if rank == JACK:
            return extract_jacks(hand_cards) != 0
        return extract_color_without_jack(hand_cards, color) != 0
    elif game_type == NULL:
        return extract_color_with_jack(hand_cards) != 0
    else: # Color
        if rank == JACK or color == game_type:
            return extract_color_trumps(hand_cards, game_type) != 0
        return extract_color_without_jack(hand_cards, color)


@njit
def get_card_strength(game_type, card_id, first_card):
    r = get_card_rank(card_id)
    if game_type == NULL:
        r = NULL_RANK_ORDER[r]
    trump = is_card_trump(game_type, card_id)
    t = 0

    if trump:
        t = 20
        if card_id in JACKS:
            t += 20 + get_card_color(card_id)
    elif get_card_color(card_id) == get_card_color(first_card):
        t = 10

    return r + t

#@njit
def get_trick_winner(game_type, cards, first_card):
    """
    Determine which cards wins the trick
    Args:
        game_type:
        cards: list of card ids
        first_card:

    Returns:
        int: index of winning card

    """
    strength_0 = get_card_strength(game_type, cards[0], first_card)
    strength_1 = get_card_strength(game_type, cards[1], first_card)
    strength_2 = get_card_strength(game_type, cards[2], first_card)
    winner = 0
    if strength_1 > strength_0:
        winner = 1
    if strength_2 > strength_1 and strength_2 > strength_0:
        winner = 2

    #print(f"Determine trick winner. game_type {game_type} cards {list(cards)} first_card {first_card} strength_0 {strength_0} strength_1 {strength_1} strength_2 {strength_2} winner {winner}")
    return winner


@njit
def calculate_game_tier(game_type, hand_cards_with_skat):
    """
    Args:
        game_type:
        hand_cards_with_skat: bitmap

    Returns: The game tier based on the Spitze (number of continuous trumps present / missing from the top). Doesn't take into account extra tiers like Schneider.

    """
    if game_type == NULL:
        return 0 # Null games don't have normal tiers

    spitze = 0
    bitmap_known_jacks = extract_jacks(hand_cards_with_skat)
    if game_type == GRAND:
        spitze = get_spitze(bitmap_known_jacks, 4)
    elif game_type < 4:
        # Color game
        bitmap_color_without_jacks = extract_color_without_jack(hand_cards_with_skat, game_type)
        spitze = get_spitze(combine_jacks_and_color(bitmap_known_jacks, bitmap_color_without_jacks), 11)
    return spitze + 1


