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
        info = episode.last_info_for()
        import pdb; pdb.set_trace()
        episode.custom_metrics["game_len"] = info["player_1"]["num_turns"]
    
    def on_train_result(self, trainer, result):
        # executes after 'timesteps_per_iteration' (1000) env steps
        pass
