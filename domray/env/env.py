import numpy as np
from collections import Counter
from ray import rllib

import domrl.engine.state as st
import domrl.engine.state_view as stv
import decision as dec
from domrl.engine.supply import BasicPiles
from domrl.engine.cards.base import BaseKingdom
# TODO: find_card_in_decision should be put in a utility.py file or something since it is shared across multiple agents
from domrl.agents.provincial_agent import find_card_in_decision
class DominionEnv(rllib.MultiAgentEnv):
    ''' Currently, I reimplement running games of dominion, but it would be
    nice if I didn't have to. I don't like to copy code.
    '''

    def __init__(self, config):
        self.config = config
        self.kingdom_and_basic_cards = sorted(BaseKingdom.keys()) + sorted(BasicPiles.keys())

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
        opponent = self.players[1 - self.current_player_idx] # NOTE: only works for 2 players
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
        
        # TODO: add in number of empty supply piles... maybe
        obs = player_obs + opponent_obs + supply_obs
    
        state_view = stv.StateView(self.state, player)
        info = state_view
        return np.array(obs), info

    def _run_until_next_buy(self, action_dict):
        # TODO: action cards that gain you stuff don't work yet
        if action_dict:
            # NOTE: we assume the action is an integer that indexes into
            # self.kingdom_and_basic_cards
            player = self.state.current_player
            decision = BuyPhaseDecision(self.state, player)
            card_name = self.kingdom_and_basic_cards[action_dict['card_idx']] # 'card_idx' must index into self.kingdom_and_basic_cards
            idx = find_card_in_decision(decision, card_name)
            decision.moves[idx].do(self.state)

        while not self.state.is_game_done():
            player = self.state.current_player
            agent = player.agent
            
            # NOTE: if/else statements copied from domrl/engine/game.py:Game.run
            if player.phase == TurnPhase.ACTION_PHASE:
                decision = ActionPhaseDecision(player)
            elif player.phase == TurnPhase.TREASURE_PHASE:
                decision = TreasurePhaseDecision(player)
            elif player.phase == TurnPhase.END_PHASE:
                decision = EndPhaseDecision(player)
            elif player.phase == TurnPhase.BUY_PHASE:
                return
            else:
                raise Exception("TurnContext: Unknown current player phase")
            
            dec.process_decision(agent, decision, self.state)

    def reset(self):
        self.state = st.GameState(agents=self.config['agents'], 
                                  players=self.config['players'], 
                                  preset_supply=config['preset_supply'],
                                  kingdoms=config['kingdoms'], 
                                  verbose=config['verbose'])
        self._run_until_next_buy(action_dict=None)
        obs, _ = self._generate_state_rep()
        return obs

    def step(self, action_dict):
        # NOTE: for now, we are guaranteed to be in the buy phase (or gain action)
        self._run_until_next_buy(action_dict)
        obs, info = self._generate_state_rep()
        
        done = self.state.is_game_done()
        if done:
            winners = self.state.get_winners()
            if len(winners) == 0:
                raise Exception("No winners despite the end of game")
            if len(winners) > 1: # TODO: figure out if ties should give any reward
                reward = 0.0
            elif player in winners:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = 0.0

        return obs, reward, done, info

        '''
        # I guess it would be nice if there was some sort of callback mechanic
        # so I didn't have to rewrite benzyx's code. So that when a buy phase
        # decision needed to be made, the game would block and send the [obs,
        # reward, done, info] information to this point right here
        '''

        



