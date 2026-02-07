from skat import get_card_color, get_card_rank, CLUBS, R_J, NULL_RANK_ORDER, GRAND, NULL, get_card_list, get_bitmap

SUIT_SYMBOLS = ["♦", "♥", "♠", "♣"]
RANK_SYMBOLS = ["7", "8", "9", "D", "K", "10", "A", "B"]
SUIT_COLORS = [33, 31, 32, 0]
COLORED_SUIT_SYMBOLS = [f"\033[{SUIT_COLORS[i]}m{SUIT_SYMBOLS[i]}\033[0m" for i in range(4)]


GAME_TYPE_NAMES = COLORED_SUIT_SYMBOLS.copy()
GAME_TYPE_NAMES.extend(['Grand', 'Null'])
EXTRA_TIER_NAMES = ['Standard', 'Hand', 'Schneider announced', 'Schwarz announced', 'Ouvert']
NULL_EXTRA_TIER_NAMES = ['', 'Hand', 'Ouvert', 'Hand Ouvert', '-']

VERBOSE_SILENT = 0
VERBOSE_PUBLIC_INFO = 1
VERBOSE_PRIVATE_INFO = 2
VERBOSE_DEBUG = 3
VERBOSE_FINEST = 4

def get_game_name(game_type, extra_tier):
    """
    Args:
        game_type (int): Color (0-3), Grand (4), Null (5)
        extra_tier (int): 0-4

    Returns:
        string:
    """
    game = GAME_TYPE_NAMES[game_type]
    if extra_tier == 0:
        return game
    extra_names = NULL_EXTRA_TIER_NAMES if game_type == NULL else EXTRA_TIER_NAMES
    extra = extra_names[extra_tier]
    return f"{game} {extra}"

def print_card_overview():
    """
    Outputs a list of all 32 cards and their ids
    .
    """
    print("Card ids:")
    print(", ".join([f"[{get_card_name(i)}] {i}" for i in range(32)]))
    print()

def get_card_name(card_id):
    """Get the textual representation of a card.

    Args:
        card_id (int): Card ID (0-31).

    Returns:
        str: Text representation of the card (e.g., "♠A").
    """
    if card_id is None or card_id < 0:
        return ""
    color_text = COLORED_SUIT_SYMBOLS[get_card_color(card_id)]
    rank_text = RANK_SYMBOLS[get_card_rank(card_id)]
    return color_text + rank_text


def color_sort_key(card_id, trump_color):
    rank = get_card_rank(card_id)
    color = get_card_color(card_id)
    segment = 100+int(color != trump_color and rank != R_J) - 10 * (rank == R_J)
    return segment, 10-color, 10-rank


def null_sort_key(card_id):
    rank = get_card_rank(card_id)
    color = get_card_color(card_id)
    return 10-color, 10-NULL_RANK_ORDER[rank]

def get_sort_key(sorting_mode):
    if sorting_mode == NULL:
        return null_sort_key
    if sorting_mode == GRAND:
        sorting_mode = CLUBS
    return lambda c: color_sort_key(c, sorting_mode)

def get_list_text(cards, sorting_mode=CLUBS):
    cards = sorted(cards, key=get_sort_key(sorting_mode))
    card_names = list(map(get_card_name, cards))
    return f"[{', '.join(card_names)}]"

def get_bitmap_text(cards, sorting_mode=CLUBS):
    """
    Convert cards (uint32 bitmask) into a human-readable string.
    Example output: "[♦7, ♥K, ♠A]"
    Args:
        cards (int): np.uint32 bitmap of length 32
        sorting_mode (int): 0-4 Colors, 5 Grand, 6 Null
    """
    card_list = get_card_list(cards)
    return get_list_text(card_list)