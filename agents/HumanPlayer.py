import numpy as np
from numpy.random.mtrand import choice

from skat import BIDDING_VALUES, get_card_list, get_bitmap
from skat_text import get_bitmap_text, GAME_TYPE_NAMES, NULL_EXTRA_TIER_NAMES, EXTRA_TIER_NAMES, color_sort_key, get_card_name, get_game_name, get_sort_key


def bid_validator(value, next_bid):
    if value == 0 or (value in BIDDING_VALUES and value >= next_bid):
        return True
    else:
        print("Kein gültiger Reizwert. Wähle einen aus", BIDDING_VALUES[BIDDING_VALUES.to_list().index(next_bid):])
        return False


class HumanPlayer():
    def __init__(self, name="Human"):
        self.name = name
        self.hand = 0
        self.position = -1
        self.game_type = None

    def get_name(self):
        return self.name

    def _get_input(self, prompt, min_val, max_val, validator=None):
        """Helper to ensure user input is valid."""
        while True:
            try:
                val = int(input(f"{prompt}: ")) # ({min_val}-{max_val})
                if min_val <= val <= max_val:
                    if validator is None or validator(val):
                        return val
                else:
                    print(f"Bitte gib eine Zahl zwischen {min_val} und {max_val} ein.")
            except ValueError:
                print("Ungültige Eingabe. Bitte verwende Zahlen von 0-9.")

    def receive_hand_cards(self, hand_cards, table_position, behaviour=1):
        self.hand = hand_cards
        self.position = table_position
        print(f"\n--- Neues Spiel ---")
        print(f"Deine Position: {['Vorhand', 'Mittelhand', 'Hinterhand'][table_position]}")
        print(f"Dein Blatt: {get_bitmap_text(hand_cards)}")


    def say(self, next_bid, history):
        """
        Sagen oder passen

            Returns:
                int: bid (18, 20, etc.) 0 means pass
        """
        return next_bid if next_bid <= self.bid else 0

    def say(self, next_bid, history):
        print(f"\nNächster möglicher Reizwert: {next_bid}")
        print(f"Blatt: {get_bitmap_text(self.hand)}")
        print(f"Gib deinen Reizwert ein oder 0 zum Passen.")
        return self._get_input("Reizwert", 0, BIDDING_VALUES[-1], lambda v: bid_validator(v, next_bid))

    def hear(self, bid, history):
        print(f"\nDu wirst gefragt: {bid}?")
        print(f"Blatt: {get_bitmap_text(self.hand)}")
        print(f"Auswahl: [0] Passen, [1] Ja")
        choice = self._get_input("Deine Wahl", 0, 1)
        return choice == 1

    def pickup_skat(self, highest_bid, history):
        print(f"Du spielst! Höchster Reizwert war {highest_bid}")
        print(f"\nBlatt: {get_bitmap_text(self.hand)}")
        print("Auswahl: [0] Handspiel (Skatkarten bleiben liegen), [1] Skat aufnehmen")
        return self._get_input("Deine Wahl", 0, 1) == 1

    def announce(self, hand_cards_with_skat):
        """Phase where player chooses game type and discards cards."""
        print(f"\nDeine Karten: {get_bitmap_text(hand_cards_with_skat)}")
        current_list = get_card_list(hand_cards_with_skat)
        skat_picked_up = len(current_list) == 12

        # 1. Put cards back int Skat
        if skat_picked_up:
            print("\nDu musst 2 Karten in den Skat legen.")
            for _ in range(2):
                # Neu sortieren für konsistente Anzeige
                current_list = sorted(current_list, key=get_sort_key(self.game_type))
                for i, c_id in enumerate(current_list):
                    print(f"[{i}] {get_card_name(c_id)} ", end="")
                print()
                drop_idx = self._get_input("Karte zum Ablegen auswählen", 0, len(current_list) - 1)
                current_list.pop(drop_idx)
        final_hand = get_bitmap(current_list)

        # 2. Choose game type
        print("Spielart wählen:")
        for i, name in enumerate(GAME_TYPE_NAMES):
            print(f"[{i}] {name}")
        g_type = self._get_input("Spielart", 0, 5)

        # 3. Choose Extra Tier
        tier = 0
        if not skat_picked_up:
            print("Stufe wählen:")
            tiers = NULL_EXTRA_TIER_NAMES if g_type == 5 else EXTRA_TIER_NAMES
            tiers = tiers[1:]
            for i, name in enumerate(tiers):
                if name != '-': print(f"[{i}] {name}")
            tier = 1 + self._get_input("Zusatzstufe", 0, len(tiers)-1)

        return g_type, tier, final_hand

    def start_playing(self, game_type, extra_tier, hand_cards, position, solo_player, ouvert_hand, bidding_history, behaviour=1):
        self.game_type = game_type
        print(f"\n--- Spielphase beginnt ---")
        print(f"Spiel: {get_game_name(game_type, extra_tier)}")
        print(f"Alleinspieler: Position {solo_player}")

    def play_card(self, hand_cards, valid_actions, current_trick, trick_giver, history):
        actions = get_card_list(valid_actions)
        cards = get_card_list(hand_cards)

        # print(f"\nStich bisher: ", end="")
        for i, card_id in enumerate(current_trick):
            if card_id != -1:
                print(f"Pos {i}: {get_card_name(card_id)}  ", end="")
        print()

        print("Handkarten:")
        cards = sorted(cards, key=get_sort_key(self.game_type))

        for i, c_id in enumerate(cards):
            print(f"[{i}] {get_card_name(c_id)}  ", end="")
        print()

        card = -1
        while card not in actions:
            choice_idx = self._get_input("Karte ausspielen", 0, len(cards) - 1)
            card = cards[choice_idx]
            if card not in actions:
                print("Du kannst diese Karte im Moment nicht ausspielen, Bedienpflicht!")
        return card