from timeit import timeit

import numpy as np

from SkatGame import SkatGame
from SkatRunner import SkatRunner
from agents.BasicAI import BasicAI
from agents.HumanPlayer import HumanPlayer
from agents.RandomAI import RandomAI
from agents.StaticAI import StaticAI
from reinforcement_learning.observation import create_obs
from skat import TRUMP_CARDS, BIDDING_VALUES
from skat_text import get_bitmap_text

obs = create_obs()
print(obs.shape)
print(BIDDING_VALUES)
quit()

runner = SkatRunner(StaticAI(), RandomAI(), BasicAI())
t = timeit(lambda: runner.run_liste(number_of_rounds=10000, equalize=True, verbosity=1, seeger_fabian=True), number=1)
print("Duration:", round(t, 3), "seconds")

#SkatGame([HumanPlayer(), BasicAI(), RandomAI()], np.random.randint(0, 3)).run()
