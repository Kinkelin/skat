from agents.bidding.RandomBiddingAI import RandomBiddingAI
from agents.playing.RandomPlayingAI import RandomPlayingAI


class RandomAI:
    def __init__(self):
        self.bidding = RandomBiddingAI()
        self.playing = RandomPlayingAI()

    def get_name(self):
        return "RandomAI"

    def receive_hand_cards(self, hand_cards, table_position, behaviour=1):
        return self.bidding.receive_hand_cards(hand_cards, table_position, behaviour)

    def say(self, next_bid, history):
        return self.bidding.say(next_bid, history)

    def hear(self, bid, history):
        return self.bidding.hear(bid, history)

    def pickup_skat(self, bid, history):
        return self.bidding.pickup_skat(bid, history)

    def announce(self, hand_cards):
        return self.bidding.announce(hand_cards)

    def start_playing(self, game_type, extra_tier, hand_cards, position, solo_player, ouvert_hand, bidding_history, behaviour=1):
        return self.playing.start_playing(game_type, extra_tier, hand_cards, position, solo_player, ouvert_hand, bidding_history, behaviour)

    def play_card(self, hand_cards, valid_actions, current_trick, trick_giver, history):
        return self.playing.play_card(hand_cards, valid_actions, current_trick, trick_giver, history)