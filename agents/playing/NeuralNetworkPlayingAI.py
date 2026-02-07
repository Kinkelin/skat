from skat import get_card_list
import torch

class NeuralNetworkPlayingAI:
    """
    PlayingAgent implementation that uses a trained PPO model.
    Only handles model calls; observation/action translation is done externally.
    """

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path (str): Path to trained PyTorch PPO model weights
            device (str or torch.device): 'cpu' or 'cuda'; auto-detect if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        if model_path is not None:
            self.load_model(model_path)
        self.last_obs = None
        self.last_action_mask = None

    def load_model(self, model_path):
        """Load a trained PyTorch model"""
        # Model class should match the one used during training
        # For example, you could have a torch.nn.Module policy
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def start_playing(self, game_type, extra_tier, hand_cards, position, solo_player, ouvert_hand, bidding_history, behaviour=1):
        """
        Called at the start of a game to set initial state.
        You can store info here if needed for observation creation.
        """
        # Store initial game info for later observation construction
        self.game_type = game_type
        self.extra_tier = extra_tier
        self.hand_cards = hand_cards
        self.position = position
        self.solo_player = solo_player
        self.ouvert_hand = ouvert_hand
        self.bidding_history = bidding_history
        self.behaviour = behaviour

    def play_card(self, hand_cards, valid_actions, current_trick, trick_giver, history):
        """
        Ask the model for the next card to play.
        Assumes external code converts the game state to the correct input tensor.
        """
        # Prepare observation (user will implement)
        obs = self._create_observation(hand_cards, valid_actions, current_trick, trick_giver, history)
        self.last_obs = obs
        self.last_action_mask = valid_actions

        # Convert observation to torch tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Forward pass through the model
            logits, _ = self.model(obs_tensor)  # Assume model returns (policy_logits, value)
            # Mask invalid actions
            logits_masked = logits.clone()
            logits_masked[0, torch.tensor([i for i, valid in enumerate(get_card_list(valid_actions)) if not valid])] = -1e9

            # Select action with highest probability
            probs = torch.softmax(logits_masked, dim=-1)
            action_index = torch.argmax(probs, dim=-1).item()

        # Convert action index back to card
        card = self._map_action_index_to_card(action_index, valid_actions)
        return card

    def _create_observation(self, hand_cards, valid_actions, current_trick, trick_giver, history):
        """
        Placeholder: user implements this
        Should return np.array of size 1183 matching training input
        """
        raise NotImplementedError("Observation creation logic must be implemented by user")

    def _map_action_index_to_card(self, action_index, valid_actions):
        """
        Placeholder: map model output index to actual card in valid_actions
        """
        raise NotImplementedError("Action mapping logic must be implemented by user")


class NeuralNetworkPlayingAI:
    """
    PlayingAgent implementation that uses a trained SB3 PPO model.
    Observation creation and mapping from model output to card is handled externally.
    """

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path (str): path to SB3 trained PPO model (.zip)
            device (str): 'cpu' or 'cuda', None will auto-detect
        """
        self.device = device
        self.model = None
        if model_path is not None:
            self.load_model(model_path)

        # Store game info for observation construction
        self.game_type = None
        self.extra_tier = None
        self.hand_cards = None
        self.position = None
        self.solo_player = None
        self.ouvert_hand = None
        self.bidding_history = None
        self.behaviour = None

    def load_model(self, model_path):
        """Load a trained SB3 PPO model"""
        self.model = PPO.load(model_path, device=self.device)

    def start_playing(self, game_type, extra_tier, hand_cards, position, solo_player, ouvert_hand, bidding_history, behaviour=1):
        """
        Called at the start of the playing phase
        """
        self.game_type = game_type
        self.extra_tier = extra_tier
        self.hand_cards = hand_cards
        self.position = position
        self.solo_player = solo_player
        self.ouvert_hand = ouvert_hand
        self.bidding_history = bidding_history
        self.behaviour = behaviour

    def play_card(self, hand_cards, valid_actions, current_trick, trick_giver, history):
        """
        Returns card selected by PPO model
        """
        if self.model is None:
            # fallback to first valid card if model not loaded
            return get_card_list(valid_actions)[0]

        # 1. Construct observation (user implements)
        obs = self._create_observation(hand_cards, valid_actions, current_trick, trick_giver, history)

        # SB3 PPO expects either 1D array or dict; convert as needed
        obs_input = np.array(obs, dtype=np.float32)

        # 2. Use SB3 PPO predict
        # deterministic=True for greedy evaluation, False for stochastic exploration
        action_index, _ = self.model.predict(obs_input, deterministic=True)

        # 3. Map action index to actual card in valid_actions
        card = self._map_action_index_to_card(action_index, valid_actions)
        return card

    def _create_observation(self, hand_cards, valid_actions, current_trick, trick_giver, history):
        """
        Placeholder: user converts game state to fixed-size input array (size 1183)
        """
        raise NotImplementedError("Observation construction must be implemented by user")

    def _map_action_index_to_card(self, action_index, valid_actions):
        """
        Placeholder: map model output index to a card in valid_actions
        """
        raise NotImplementedError("Action mapping must be implemented by user")