import numpy as np
from agents.bidding.BasicBiddingAI import BasicBiddingAI, calculate_bid
from tabulate import tabulate
from time import time
from skat import add_skat_to_hand, get_bitmap, NULL, deal_new_cards_from_deck, NUMBER_OF_CARDS
from skat_text import get_bitmap_text, GAME_TYPE_NAMES, EXTRA_TIER_NAMES, NULL_EXTRA_TIER_NAMES, print_card_overview, get_game_name


def simulate_bidding(agent, cards, skat, table_position, verbose=True, comment=None, risk_taking=1, bidding_war=np.random.rand()):
    if comment is not None:
        print(comment+":")
    if verbose:
        print(f"Hand dealt: {get_bitmap_text(cards)} ({cards}, {skat})")
    agent.receive_hand_cards(cards, table_position, risk_taking)
    bid = agent.simplified_bidding(cards)
    passed = bid == 0
    if passed:
        game_type = -1
        extra_tier = -1
        if verbose:
            print("Agent passes\n")
    else:
        if verbose:
            print("Agent max bid", bid)
        final_bid = 18 + int(round(bidding_war * (bid-18)))
        use_skat = agent.pickup_skat(final_bid, None)
        if use_skat:
            if verbose:
                print("Agent picks up Skat:", get_bitmap_text(skat))
            cards = add_skat_to_hand(cards, skat)
        elif verbose:
            print("Agent announces Hand")
        game_type, extra_tier, hand = agent.announce(cards)
        if verbose:
            print(f"Agent plays {get_game_name(game_type, extra_tier)}")
            print(f"Cards: {get_bitmap_text(hand, game_type)} ({ hand })\n")

    return passed, game_type, extra_tier

def run_biddings(agent, nr_of_biddings, risk_taking=1, rng_seed=12345):
    print(f"Simuliere {nr_of_biddings} Ansagen (risk_taking {risk_taking}, seed {rng_seed})")
    rng = np.random.default_rng(rng_seed)
    column_names = GAME_TYPE_NAMES
    game_types = np.zeros(6, dtype=np.float64)
    extra_tiers = np.zeros((6,5), dtype=np.float64)
    passes = 0
    start = time()
    # Generate random data
    rand_vals = rng.random(size=(nr_of_biddings, NUMBER_OF_CARDS), dtype=np.float32)
    shuffled_decks = np.argsort(rand_vals, axis=1)
    bidding_wars = rng.random(size=nr_of_biddings, dtype=np.float32)
    for i in range(nr_of_biddings):
        cards_p0, cards_p1, cards_p2, skat = deal_new_cards_from_deck(shuffled_decks[i,:])
        table_position = i % 3
        passed, game_type, extra_tier = simulate_bidding(agent, cards_p0, skat, table_position, False, None, risk_taking, bidding_wars[i])
        if passed:
            passes += 1
        else:
            game_types[game_type] += 1
            extra_tiers[game_type, extra_tier] += 1
    simulation_time = time() - start
    print(f"\nGepasst:\n{(passes * 100) / nr_of_biddings}%")
    not_passed = nr_of_biddings - passes
    game_types *= 100/not_passed
    data = [[str(round(g, 1)) + "%" for g in game_types]]
    print("\nSpielarten Ansagehäufigkeit:")
    print(tabulate(data, headers=column_names))
    print()
    data2 = [ [EXTRA_TIER_NAMES[r]] + [str(round(e, 1)) for e in extra_tiers[:, r]] + [NULL_EXTRA_TIER_NAMES[r]] for r in range(extra_tiers.shape[1]) ]
    data2[0][0] = "Standard"
    print(tabulate(data2, headers=["Ansage"] + column_names + ["Nullspiel"]))
    print(f"\nTime:\n{round(simulation_time, 3)}\n")


def main():
    print_card_overview()
    agent = BasicBiddingAI()

    # Example calls for testing:
    print("Beispiel 1:")
    simulate_bidding(agent, 154255584, 4325376, 0, True,"Hier sollte man auf Grand reizen und nach Aufnehmen des Skats die beiden Kreuze drücken")

    example_null_hand = get_bitmap([0, 1, 18, 8, 9, 24, 26, 28, 7, 31])  # This is a solid null hand, with only a lonely ♠9 as weakness
    bid, game_type, extra_tier, confidence, use_skat = calculate_bid(example_null_hand)  # bid, game_type, np.uint32(extra_tier), confidence, use_skat
    print(f"Beispiel 2:\ncalculate_bid({get_bitmap_text(example_null_hand, NULL)}) -> bid: {bid} game: {get_game_name(game_type, extra_tier)} confidence: {round(confidence, 3)} use_skat: {use_skat}\n")

    # Run full simulations
    run_biddings(agent, 10_000, 0.8)
    run_biddings(agent, 10_000)
    run_biddings(agent, 10_000, 1.25)

    run_biddings(agent, 1_000_000)

if __name__ == "__main__":
    main()