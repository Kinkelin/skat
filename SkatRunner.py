import itertools

import numba
from tabulate import tabulate
import numpy as np

from SkatGame import SkatGame
from skat import NUMBER_OF_CARDS, deal_new_cards, deal_new_cards_as_bitmaps, RESULT_SOLO_WIN, RESULT_TEAM_WIN, RESULT_PASSED
from skat_text import VERBOSE_PUBLIC_INFO, VERBOSE_SILENT


class SkatRunner:

    def __init__(self, player0, player1, player2):
        self.players = np.array([player0, player1, player2])
        self.verbosity = 1
        self.point_total = np.zeros(3, dtype=np.int64)
        self.solo = np.zeros(3, dtype=np.int64)
        self.solo_wins = np.zeros(3, dtype=np.int64)
        self.passed_games = 0
        self.team = np.zeros(3, dtype=np.int64)
        self.team_wins = np.zeros(3, dtype=np.int64)
        self.seeger_fabian = True
        self.number_of_rounds = 3
        self.equalize = False
        self.games_played = 0

    def run_liste(self, number_of_rounds=3, equalize=False, verbosity=1, seeger_fabian=True):
        """
        Args:
            number_of_rounds (int): Games to be played. With the equalize option, 6 times this will be the number of games
            equalize (bool): Plays every game 6 times, rotating the players around
            verbosity (int): How much information should be printed out
            seeger_fabian (bool): If Seeger Fabian modifiers should be applied to game result
        """
        self.number_of_rounds = number_of_rounds
        self.equalize = equalize
        self.verbosity = verbosity
        self.seeger_fabian = seeger_fabian
        self.games_played = 0
        self.passed_games = 0

        if verbosity >= 1:
            print(f"Play {number_of_rounds} rounds with {self.players[0].get_name()}, {self.players[1].get_name()} and {self.players[2].get_name()}. Equalize: {equalize}")

        forehand = 0 # Vorhand
        for i in range(number_of_rounds):
            if equalize:
                cards = deal_new_cards_as_bitmaps()
                behaviours = np.random.rand(3) * 0.2 + 0.9
                player_ids = np.arange(3)
                for perm in itertools.permutations(player_ids):
                    inv_perm = np.argsort(np.array(perm))
                    game = SkatGame(self.players[np.array(perm)].copy(), forehand, cards.copy(), VERBOSE_SILENT, behaviours)
                    game_result, game_points = game.run()
                    game_points = game_points[inv_perm]
                    self.process_game(game_result, game_points)
            else:
                game = SkatGame(self.players, forehand, None, VERBOSE_SILENT)
                game_result, game_points = game.run()
                self.process_game(game_result, game_points)

            if verbosity >= 2:
                print(f"{self.games_played} Spiele gespielt. Zwischenstand: {self.point_total}")

            forehand = (forehand + 1) % 3

        if verbosity >= 1:
            self.print_results()

    def process_game(self, game_result, game_points):
        if game_result == RESULT_PASSED:
            self.passed_games += 1
        else:
            self.solo[game_points != 0] += 1
            self.team[game_points == 0] += 1
            if game_result == RESULT_SOLO_WIN:
                self.solo_wins[game_points != 0] += 1
            elif game_result == RESULT_TEAM_WIN:
                self.team_wins[game_points == 0] += 1
            self.apply_seeger_fabian(game_result, game_points)
            self.point_total += game_points
        self.games_played += 1

    def apply_seeger_fabian(self, game_result, game_points):
        if self.seeger_fabian:
            if game_result == RESULT_SOLO_WIN:
                game_points[game_points > 0] += 50
            elif game_result == RESULT_TEAM_WIN:
                game_points[game_points == 0] += 40
                game_points[game_points < 0] -= 50
            if self.verbosity >= 2:
                print(f"Seeger-Fabian Wertung angewendet. Punkte für diese Runde: {game_points}")



    def print_results(self):
        print(f"Games passed by all players: {round(100 * self.passed_games / self.games_played)}%")
        data = []
        for i, points in enumerate(self.point_total):
            data.append([self.players[i].get_name(), int(points), round(points / self.games_played, 2), f"{round(100 * self.solo[i] / self.games_played, 2)}%", f"{round(100 * self.solo_wins[i] / self.solo[i], 2) if self.solo[i] > 0 else 0}%", f"{round(100 * self.team_wins[i] / self.team[i], 2) if self.team[i] > 0 else 0}%"])
        headers = ['Player', 'Score', 'Avg per game', 'Solo', 'Wins solo', 'Wins team']
        print()
        print(tabulate(data, headers=headers))
        print()






