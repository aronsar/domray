from typing import Dict
import argparse
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class DomCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        info = episode._agent_to_last_info
        episode.hist_data['game_len'] = [max(info['player_1']['num_turns'], info['player_2']['num_turns'])]
        for player in info:
            for metric in info[player]:
                if isinstance(info[player][metric], dict):
                    for m in info[player][metric]:
                        key = player + '_' + metric + '_' + m
                        episode.hist_data[key] = [info[player][metric][m]]
                else:
                    key = player + '_' + metric
                    episode.hist_data[key] = [info[player][metric]]
        pass
    
    def on_train_result(self, trainer, result):
        pass
        # executes after 'timesteps_per_iteration' (1000) env steps
        # if step % check_point_freq == 0:
            # evaluate against provincial buy menu
            # save raw games and metrics
            # 
            # 
            # 
            # 
