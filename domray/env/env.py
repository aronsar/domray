import numpy as np
import copy
from collections import Counter
from ray import rllib
import gym
from gym.spaces import Box, Dict, Discrete

import domrl.engine.state as st
import domrl.engine.state_view as stv
import domrl.engine.decision as dec
from domrl.engine.util import TurnPhase
from domrl.engine.supply import BasicPiles
from domrl.engine.cards.base import BaseKingdom
from domrl.engine.decision import EndBuyPhase
# TODO: find_card_in_decision should be put in a utility.py file or something since it is shared across multiple agents
from domrl.agents.provincial_agent import find_card_in_decision


class DominionEnv(gym.Env):
    ''' Currently, I reimplement running games of dominion, but it would be
    nice if I didn't have to. I don't like to copy code.
    '''

    def __init__(self, config):
        self.config = config
        self.kingdom_and_basic_cards = sorted(BaseKingdom.keys()) + sorted(BasicPiles.keys())
        self.max_avail_actions = len(self.kingdom_and_basic_cards) + 1
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.max_avail_actions, )),
            "state": Box(low=0.0, high=55.0, shape=(100,), dtype=np.float32)
        })
        self.action_space = Discrete(self.max_avail_actions)
        self.reward_range = (-1.0, 1.0)


    def _generate_state_rep(self):
        ''' Very simple state representation; not scalable to multiple players.
        The representation is a concatenation of three vectors. Each of these
        three is just the count of how many cards there are in your deck, your
        opponent's deck, and the supply piles. "Deck" in this context includes
        your deck, your discard, and your hand (ie. all your cards).
        '''

        player_obs = []
        player = self.state.current_player
        player_card_count = Counter([card.name for card in player.all_cards])
        for card in self.kingdom_and_basic_cards:
            if card in self.state.supply_piles:
                player_obs.append(player_card_count[card])
            else:
                player_obs.append(0)
        
        opponent_obs = []
        opponent_idx = self.state.next_player_idx(self.state.current_player_idx)
        opponent = self.state.players[opponent_idx] # NOTE: only works for 2 players
        opponent_card_count = Counter([card.name for card in opponent.all_cards])
        for card in self.kingdom_and_basic_cards:
            if card in self.state.supply_piles:
                opponent_obs.append(opponent_card_count[card])
            else:
                opponent_obs.append(0)

        supply_obs = []
        for card in self.kingdom_and_basic_cards:
            if card in self.state.supply_piles:
                supply_obs.append(self.state.supply_piles[card].qty)
            else:
                supply_obs.append(0)
        
        num_empty_piles = 0
        for card in self.state.supply_piles:
            if self.state.supply_piles[card].qty == 0:
                num_empty_piles += 1
        
        obs = player_obs + opponent_obs + supply_obs + [num_empty_piles]
         
        state_view = stv.StateView(self.state, player)
        info = {} # TODO: add something like dict(state_view)
        return np.array(obs, dtype=np.float32), info

    def _run_until_next_buy(self, action):
        # TODO: action cards that gain you stuff don't work yet
        if action:
            # NOTE: we assume the action is an integer that indexes into
            # self.kingdom_and_basic_cards
            player = self.state.current_player
            decision = dec.BuyPhaseDecision(self.state, player)
            if action == len(self.kingdom_and_basic_cards):
                idx = 0
                # TODO: refactor stats logging so it meshes with existing logging
                player.stats['wasted_coins'].append(player.coins)
            else:
                card_name = self.kingdom_and_basic_cards[action]
                idx = find_card_in_decision(decision, card_name)[0]
                if idx == 0:
                    import pdb; pdb.set_trace()
            
            decision.moves[idx].do(self.state)

        while not self.state.is_game_over():
            player = self.state.current_player
            agent = player.agent
            
            # NOTE: if/else statements copied from domrl/engine/game.py:Game.run
            if player.phase == TurnPhase.ACTION_PHASE:
                decision = dec.ActionPhaseDecision(player)
            elif player.phase == TurnPhase.TREASURE_PHASE:
                decision = dec.TreasurePhaseDecision(player)
            elif player.phase == TurnPhase.END_PHASE:
                decision = dec.EndPhaseDecision(player)
            elif player.phase == TurnPhase.BUY_PHASE:
                return
            else:
                raise Exception("TurnContext: Unknown current player phase")
            
            dec.process_decision(agent, decision, self.state)

    def _print_stats(self):
        def print_scores(players):
            for player in players:
                print('Player {} score: {}'.format(player.idx+1, player.total_vp()))

        def print_buy_stats(players):
            for player in players:
                w = player.stats['wasted_coins']
                print('Player {} wasted (min, avg, max, num) coins: ({}, {:.1f}, {})'.format(player.idx+1, min(w), sum(w)/len(w), max(w), len(w)))

        def print_deck_comps(players):
            for player in players:
                print('----> Deck Composition of Player {} <----'.format(player.idx+1))

                deck = Counter()
                for card in player.all_cards:
                    deck[card.name] += 1
                for cardname in sorted(deck):
                    print('{:<15}: {}'.format(cardname, deck[cardname]))

        print('*****************************************************************************')
        print('Number of turns to finish game: ' + str(self.state.turn))
        print_scores(self.state.players)
        print_buy_stats(self.state.players)
        print_deck_comps(self.state.players)
        print('*****************************************************************************')

    def _action_mask(self):
        player = self.state.current_player
        decision = dec.BuyPhaseDecision(self.state, player)
        action_mask = np.zeros(self.max_avail_actions, dtype=np.float32)
        buyable_cards = []

        for choice in decision.moves:
            if isinstance(choice, EndBuyPhase):
                action_mask[-1] = 1.0
            else:
                buyable_cards.append(choice.card_name)

        for card_idx, cardname in enumerate(self.kingdom_and_basic_cards):
            if cardname in buyable_cards:
                action_mask[card_idx] = 1.0

        return action_mask

    def reset(self):
        preset_supply = copy.deepcopy(self.config['preset_supply'])
        self.state = st.GameState(agents=self.config['agents'], 
                                  players=self.config['players'], 
                                  preset_supply=preset_supply,
                                  kingdoms=self.config['kingdoms'], 
                                  verbose=self.config['verbose'])
        self._run_until_next_buy(action=None)
        obs, _ = self._generate_state_rep()
        obs = {
            'action_mask': self._action_mask(),
            'state': obs
        }
        return obs

    def step(self, action):
        # TODO: refactor all this logging/stats printing code elsewhere

        # NOTE: for now, we are guaranteed to be in the buy phase (or gain action)
        self._run_until_next_buy(action)
        obs, info = self._generate_state_rep()
        
        done = self.state.is_game_over()
        if done:
            self._print_stats()            
            winners = self.state.get_winners()
            if len(winners) == 0:
                raise Exception("No winners despite the end of game")
            if len(winners) > 1: # TODO: figure out if ties should give any reward
                reward = 0.0
            elif self.state.current_player in winners:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = 0.0
        
        obs = {
            'action_mask': self._action_mask(),
            'state': obs
        }
        return obs, reward, done, info

        '''
        # I guess it would be nice if there was some sort of callback mechanic
        # so I didn't have to rewrite benzyx's code. So that when a buy phase
        # decision needed to be made, the game would block and send the [obs,
        # reward, done, info] information to this point right here

        NOTE: there is, use rllib.env.ExternalEnv

        '''

        



