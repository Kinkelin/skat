from numbers import Number

import numpy as np

from skat import deal_new_cards, BIDDING_VALUES, deal_new_cards_as_bitmaps, add_skat_to_hand, get_cards_that_have_been_removed, count_points, get_valid_actions, get_trick_winner, get_card_points, \
    NULL, EXTRA_TIER_SCHWARZ, EXTRA_TIER_SCHNEIDER, calculate_game_tier, BIDDING_BASE_VALUES, BIDDING_NULL, get_bitmap, remove_card, is_card_present, count_cards, EXTRA_TIER_OUVERT, \
    EXTRA_TIER_NULL_OUVERT, EXTRA_TIER_NULL_HAND_OUVERT, RESULT_SOLO_WIN, RESULT_TEAM_WIN, RESULT_PASSED
from skat_text import get_card_name, get_game_name, get_bitmap_text, VERBOSE_PUBLIC_INFO, VERBOSE_PRIVATE_INFO, get_list_text


class SkatGame:

    def __init__(self, players, forehand, cards=None, verbosity=VERBOSE_PUBLIC_INFO, behaviours=np.array([1, 1, 1])):
        """

        Args:
            players: list of 3 instances of SkatPlayer-like classes
            forehand: 0-2 index of forehand player in players array
            cards: np array [player0_cards, player1_cards, player2_cards, skat]
            verbose: bool
        """
        self.players = list(players)
        self.cards = cards
        self.solo_cards = 0
        self.forehand = forehand  # index of player in self.players
        self.middlehand = (forehand + 1) % 3  # index of player in self.players
        self.rearhand = (forehand + 2) % 3  # index of player in self.players
        self.verbosity = verbosity
        self.behaviours = behaviours
        self.highest_bid = 0
        self.highest_bidder = None # player instance
        self.history = None
        self.game_type = 0
        self.extra_tier = 0
        self.solo_player = 0 # index of player in self.players

        if self.verbose():
            print(f"Beginne Spiel mit {self.get_player_text(0)} {self.get_player_text(1)} {self.get_player_text(2)}. Vorhand: {self.get_player_text(self.forehand)} Verhalten: {self.behaviours}")

        if self.cards is None:
            # Deal cards, 10 to each player and 2 to the skat
            self.cards = deal_new_cards_as_bitmaps()
            if self.verbose(VERBOSE_PRIVATE_INFO):
                print(f"Karten ausgeteilt:\n{self.get_player_text(0)} {get_bitmap_text(self.cards[0])}\n{self.get_player_text(1)} {get_bitmap_text(self.cards[1])}\n{self.get_player_text(2)} {get_bitmap_text(self.cards[2])}\nSkat {get_bitmap_text(self.cards[3])}")
        self.start_cards = self.cards.copy()
        self.solo_win = False
        self.schneider = False
        self.schwarz = False
        self.points = np.zeros(3, dtype=np.int64)
        self.tricks = np.full((10, 3), -1, dtype=np.int8)

    def verbose(self, verbosity=VERBOSE_PUBLIC_INFO):
        return self.verbosity >= verbosity

    def run(self):
        """

        Returns:
            tuple: (result, points) RESULT_ constant, np.array size 3 int

        """
        self.highest_bid, self.highest_bidder = self.bidding()
        if self.highest_bid > 0:
            self.solo_player = self.players.index(self.highest_bidder)
            pickup_skat = self.highest_bidder.pickup_skat(self.highest_bid, self.history)
            if self.verbose():
                print(f"{self.get_player_text(self.highest_bidder)} hat mit {self.highest_bid} das höchste Gebot abgegeben und sagt an.")
                pickup_text = "" if pickup_skat else "nicht "
                print(f"{self.get_player_text(self.highest_bidder)} nimmt den Skat {pickup_text}auf.")
            hand_cards = self.cards[self.solo_player]
            original_hand_cards = hand_cards
            self.solo_cards = add_skat_to_hand(hand_cards, self.cards[3]) # Used for points calculation
            if pickup_skat:
                hand_cards = self.solo_cards
            self.game_type, self.extra_tier, self.cards[self.solo_player] = self.highest_bidder.announce(hand_cards)
            if (not pickup_skat) and original_hand_cards != self.cards[self.solo_player]:
                raise DirtyCheatingError(self.get_player_text(self.solo_player),
                                         f"Hat den Skat nicht aufgenommen, aber trotzdem andere Handkarten. Vorher: {get_bitmap_text(original_hand_cards)} Nachher: {get_bitmap_text(self.cards[self.solo_player])}")
            if pickup_skat:
                self.cards[3] = get_cards_that_have_been_removed(hand_cards, self.cards[self.solo_player])
                if count_cards(self.cards[self.solo_player]) != 10 or count_cards(self.cards[3]) != 2:
                    raise DirtyCheatingError(self.get_player_text(self.solo_player), f"Hat nicht ordentlich Karten in den Skat zurückgelegt. Handkarten inkl. Skat {get_bitmap_text(hand_cards)} Handkarten nachher {get_bitmap_text(self.cards[self.solo_player])} Skat neu {get_bitmap_text(self.cards[3])}")
                if self.verbose(VERBOSE_PRIVATE_INFO):
                    print(f"{self.get_player_text(self.highest_bidder)} legt {get_bitmap_text(self.cards[3])} zurück in den Skat.")
            if self.verbose():
                print(f"{self.get_player_text(self.highest_bidder)} spielt {get_game_name(self.game_type, self.extra_tier)}")
            self.playing()
            self.calculate_points()
            if self.verbose():
                print("Spiel vorbei.")
            result = RESULT_SOLO_WIN if self.solo_win else RESULT_TEAM_WIN
        else:
            if self.verbose():
                print("Alle haben gepasst, Spiel vorbei.")
            result = RESULT_PASSED
        if self.verbose():
            print(f"Kartenverteilung war {self.get_player_text(self.players[0])}: {get_bitmap_text(self.start_cards[0])} {self.get_player_text(self.players[1])}: {get_bitmap_text(self.start_cards[1])} {self.get_player_text(self.players[2])}: {get_bitmap_text(self.start_cards[2])} Skat: {get_bitmap_text(self.start_cards[3])}")
        return result, self.points.copy()

    def get_player_text(self, player):
        if isinstance(player, Number):
            player = self.players[player]
        return f"{player.get_name()} ({self.players.index(player)})";

    def print_passed(self, player):
        if self.verbose():
            print(f"{self.get_player_text(player)} passt.")

    def print_say(self, player, player_bid):
        if self.verbose():
            print(f"{self.get_player_text(player)} sagt {player_bid}.")

    def print_hear(self, player):
        if self.verbose():
            print(f"{self.get_player_text(player)} sagt Ja.")


    def bidding(self):
        """

        Returns: highest_bid, highest_bidder (Everyone passed -> highest_bid 0)

        """
        for i in range(3):
            self.players[i].receive_hand_cards(self.cards[i], (3+i-self.forehand) % 3, self.behaviours[i]) # Players receive relative positions

        fore = self.players[self.forehand]
        middle = self.players[self.middlehand]
        rear = self.players[self.rearhand]

        saying = middle
        hearing = fore
        highest_bid = 0
        highest_bidder = None

        i = 0
        while i < len(BIDDING_VALUES):
            bid = BIDDING_VALUES[i]
            player_bid = saying.say(bid, self.history)
            if player_bid < bid or player_bid not in BIDDING_VALUES:
                # Passed
                self.print_passed(saying)
                if saying == middle:
                    saying = rear
                else:
                    # No more saying, bidding is over
                    break
            else:
                self.print_say(saying, player_bid)
                highest_bid = player_bid
                highest_bidder = saying
                accepted = hearing.hear(player_bid, self.history)
                if accepted:
                    self.print_hear(hearing)
                    highest_bidder = hearing
                else:
                    self.print_passed(hearing)
                    if saying == middle:
                        hearing = middle
                        saying = rear
                    else:
                        # Bidding is over
                        break
                i += 1

        if highest_bid == 0:
            player_bid = fore.say(18, self.history)
            if player_bid in BIDDING_VALUES:
                self.print_say(fore, player_bid)
                highest_bid = player_bid
                highest_bidder = fore
            else:
                self.print_passed(fore)

        return highest_bid, highest_bidder


    def play_card(self, player, leader, trick):
        """
        Handles playing of a card
        Args:
            player: Index of player in self.players
            leader: Index of leader
            trick: Index of trick (0-9)

        """
        hand_cards = self.cards[player]
        valid_actions = hand_cards if player == leader else get_valid_actions(self.game_type, self.tricks[trick, leader], hand_cards)

        #print(self.get_player_text(player), "hand_cards", get_bitmap_text(hand_cards), "valid_actions", get_bitmap_text(valid_actions), "current_trick", get_list_text(self.tricks[trick]))
        card = self.players[player].play_card(
            hand_cards,
            valid_actions,
            self.tricks[trick],
            leader,
            self.history
        )
        if not is_card_present(hand_cards, card):
            raise DirtyCheatingError(self.get_player_text(player), f"Hat {get_card_name(card)} aus dem Ärmel gezaubert, bei diesen Handkarten: {get_bitmap_text(hand_cards)}")
        if not is_card_present(valid_actions, card):
            raise DirtyCheatingError(self.get_player_text(player), f"Hat {get_card_name(card)} aus gespielt, obwohl nur diese Optionen erlaubt waren: {get_bitmap_text(valid_actions)}")
        self.tricks[trick, player] = card
        self.cards[player] = remove_card(hand_cards, card)
        if self.verbose():
            print(f"{self.get_player_text(player)} spielt {get_card_name(card)}")

    def playing(self):

        self.solo_win = False
        self.schneider = False
        self.schwarz = False

        leader = self.forehand
        ouvert = (self.game_type != NULL and self.extra_tier == EXTRA_TIER_OUVERT) or (self.game_type == NULL and self.extra_tier == EXTRA_TIER_NULL_OUVERT) or (self.game_type == NULL and self.extra_tier == EXTRA_TIER_NULL_HAND_OUVERT)
        ouvert_hand = self.cards[self.solo_player] if ouvert else 0

        for i in range(3):
            self.players[i].start_playing(self.game_type, self.extra_tier, self.cards[i].copy(), i, self.solo_player, ouvert_hand, self.history, self.behaviours[i])

        solo_points = count_points(self.cards[3])
        team_points = 0
        for i in range(10):
            second = (leader + 1) % 3
            third = (leader + 2) % 3
            if self.verbose(VERBOSE_PRIVATE_INFO):
                print(f"Stich {i+1} Handkarten: {self.get_player_text(0)} {get_bitmap_text(self.cards[0], self.game_type)}     {self.get_player_text(1)} {get_bitmap_text(self.cards[1], self.game_type)}     {self.get_player_text(2)} {get_bitmap_text(self.cards[2], self.game_type)}")
            self.play_card(leader, leader, i)
            self.play_card(second, leader, i)
            self.play_card(third, leader, i)
            # play_card(hand_cards, valid_actions, current_trick, trick_giver, history)
            winner = get_trick_winner(self.game_type, self.tricks[i], self.tricks[i, leader])
            if self.verbose():
                print(f"Stich geht an {self.get_player_text(winner)}")
            points = count_points(get_bitmap(self.tricks[i]))

            if winner == self.solo_player:
                solo_points += points
                if self.game_type == NULL:
                    # Solo player loses upon getting a trick in a Null game
                    self.solo_win = False
                    if self.verbose():
                        print("Alleinspieler hat einen Stich bekommen: Nullspiel verloren.")
                    return
            else:
                team_points += points
                if self.game_type != NULL and self.extra_tier >= EXTRA_TIER_SCHWARZ:
                    # Solo player loses upon giving up a trick in a Schwarz or Ouvert game
                    self.solo_win = False
                    if self.verbose():
                        print("Gegenpartei hat einen Stich bekommen: Schwarz kann nicht mehr erreicht werden.")
                    return
            leader = winner

        if self.verbose():
            print("Alle Karten wurden gespielt.")
        if self.game_type == NULL:
            # Solo player in a Null game wins for not getting a single trick
            self.solo_win = True
            if self.verbose():
                print("Alleinspieler hat keinen einzigen Stich bekommen: Nullspiel gewonnen")
            return

        self.schneider = team_points <= 30
        self.schwarz = team_points == 0
        if self.verbose():
            print(f"Augen - Alleinspieler: {solo_points} Gegenpartei: {team_points}")
            if self.schneider:
                print("Schneider erreicht")
            if self.schwarz:
                print("Schwarz erreicht")
        if self.extra_tier == EXTRA_TIER_SCHNEIDER:
            self.solo_win = self.schneider
        else:
            self.solo_win = solo_points > 60
        if self.verbose():
            if self.solo_win:
                print("Alleinspieler gewinnt")
            else:
                print("Gegenpartei gewinnt")
        return


    def calculate_points(self):
        if self.game_type == NULL:
            game_points = np.int32(BIDDING_NULL[self.extra_tier])
            if self.verbose():
                print(f"Spielwert {get_game_name(NULL, self.extra_tier)}: {game_points} Punkte")
        else:
            tier = calculate_game_tier(self.game_type, self.solo_cards)
            extra_tier = self.extra_tier
            if self.solo_win:
                if self.schneider:
                    extra_tier += 1
                if self.schwarz:
                    extra_tier += 1
            game_points = np.int32(BIDDING_BASE_VALUES[self.game_type] * (tier + extra_tier))

            if self.verbose():
                print(f"Spielwert {get_game_name(self.game_type, self.extra_tier)}: Farbwert {BIDDING_BASE_VALUES[self.game_type]} Stufe {tier} extra {extra_tier} Punkte gesamt {game_points}")

        if (not self.solo_win) or game_points < self.highest_bid:
            game_points *= -2

        if self.verbose():
            print(f"Alleinspieler {self.get_player_text(self.solo_player)} bekommt {game_points} Punkte")

        self.points[self.solo_player] = game_points


class DirtyCheatingError(Exception):

    def __init__(self, player, violation):
        super().__init__(f"{player} schummelt: {violation}")