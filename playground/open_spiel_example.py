import pyspiel
import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import ppo

# 1. Setup the Game
# We disable the double dummy solver to force agents to play the cards out manually.
game_name = "bridge"
game_config = {"use_double_dummy_result": False}

# Create the environment
env = rl_environment.Environment(game_name, **game_config)
num_players = env.num_players
num_actions = env.action_spec()["num_actions"]
info_state_shape = env.observation_spec()["info_state"]

# 2. Setup PPO Agents
# We create 4 independent agents (one for each seat: North, East, South, West).
agents = [
    ppo.PPO(
        input_shape=info_state_shape,
        num_actions=num_actions,
        num_players=num_players,
        player_id=idx,
        num_envs=1,
        steps_per_batch=128,  # Batch size for learning
        num_minibatches=4,
        update_epochs=4,
        learning_rate=3e-4,
        gae=True,
        gamma=0.99,
        clip_coef=0.2,
        entropy_coef=0.01,
    )
    for idx in range(num_players)
]

# 3. Training Loop
print("Starting training...")
for episode in range(100):
    time_step = env.reset()

    while not time_step.last():
        current_player = time_step.observations["current_player"]

        # Get action from the agent whose turn it is
        # The agent helper automatically handles legal action masking
        agent_output = agents[current_player].step(time_step)

        # Step the environment
        time_step = env.step([agent_output.action])

    # 4. Learning Step (at end of episode)
    # Bridge is zero-sum, so agents learn from the final reward.
    for agent in agents:
        agent.step(time_step)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1} finished. Last rewards: {time_step.rewards}")

print("Training complete.")