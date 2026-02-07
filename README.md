# A python implementation of the card game Skat

This repository is currently very much work in progress. You can run play.py or bidding_simulation.py to see it in action.

## The goal
The ultimate aim of the project is to create strong, open source, Skat AI and make it available in a GUI for turn by turn game analysis, similar to tools available with chess engines. 

Evaluating from the perspective of the individual players with imperfect information. 

This would include win probabilities for bidding options, engine evaluation of card options in trick play and expected game outcome. 

## Roadmap
The current focus is currently is to implement a good base for AI implementations and training.

- ✔ Game logic base: skat.py defines constants and basic game logic, including a number of numba.njit compiling functions to enable fast simulations.
- ✔ Agent interfaces: SkatPlayer includes all methods to receive game information and return player decisions. The game can be split into a bidding and a playing phase, with the BiddingAgent and PlayingAgent interfaces.
- ✔ Realistic bidding AI: BasicBiddingAI uses an algorithmic approach to achieve acceptable bidding results. Should be good enough to create varied and playable game setups for reinforcement learning of playing phase AI.
- ✔ Agent implementations: BasicAI, RandomAI, StaticAI and HumanPlayer
- ✔ Game implementation: The SkatGame class implements a full game of Skat.
- ✔ Skat Listen: Simulate a large number of games with the SkatRunner class to compare player strength.
- ✔ Observation space (Playing phase): observation.py defines an observation space for reinforcement learning of playing phase agents.
- WIP Environment (Playing phase): SkatPlayingEnv is a gymnasium environment for reinforcement learning of playing phase agents.
- ✖ Train playing phase agent using PPO.
- ✖ Train bidding phase agent utilizing fully played out games with a strong playing phase agent.
- ✖ Evaluate playing strength against real human players. Ideally through collaboration with an existing online Skat playerbase.
- ✖ Create GUI for AI supported game analysis
