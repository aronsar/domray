import numpy as np
import copy
from collections import Counter
from ray import rllib
import gym
from gym.spaces import Box, Discrete

import domrl.engine.state as st
import domrl.engine.state_view as stv
import domrl.engine.decision as dec
from domrl.engine.util import TurnPhase
from domrl.engine.supply import BasicPiles
from domrl.engine.cards.base import BaseKingdom
# TODO: find_card_in_decision should be put in a utility.py file or something since it is shared across multiple agents
from domrl.agents.provincial_agent import find_card_in_decision


#debug only
from domrl.engine.util import TurnPhase

class DominionEnv(gym.Env):
    ''' Currently, I reimplement running games of dominion, but it would be
    nice if I didn't have to. I don't like to copy code.
    '''

    def __init__(self, config):
        self.config = config
        self.kingdom_and_basic_cards = sorted(BaseKingdom.keys()) + sorted(BasicPiles.keys())
        self.observation_space = Box(low=0.0, high=55.0, shape=(1,100), dtype=np.float32)
        self.action_space = Discrete(len(self.kingdom_and_basic_cards) + 1)
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
        
        # TODO: add in number of empty supply piles... maybe
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
            else:
                card_name = self.kingdom_and_basic_cards[action]
                idx = find_card_in_decision(decision, card_name)[0]

            if not self.state.current_player.phase == TurnPhase.BUY_PHASE:
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

    def reset(self):
        preset_supply = copy.deepcopy(self.config['preset_supply'])
        self.state = st.GameState(agents=self.config['agents'], 
                                  players=self.config['players'], 
                                  preset_supply=preset_supply,
                                  kingdoms=self.config['kingdoms'], 
                                  verbose=self.config['verbose'])
        self._run_until_next_buy(action=None)
        obs, _ = self._generate_state_rep()
        return [obs]

    def step(self, action):
        # NOTE: for now, we are guaranteed to be in the buy phase (or gain action)
        self._run_until_next_buy(action)
        obs, info = self._generate_state_rep()
        
        done = self.state.is_game_over()
        if done:
            print('**************************************************************************************************************')
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

        return [obs], reward, done, info

        '''
        # I guess it would be nice if there was some sort of callback mechanic
        # so I didn't have to rewrite benzyx's code. So that when a buy phase
        # decision needed to be made, the game would block and send the [obs,
        # reward, done, info] information to this point right here

        NOTE: there is, use rllib.env.ExternalEnv

        '''

        



