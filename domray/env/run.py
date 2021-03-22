import ray
import torch
import numpy as np
from gym.spaces import Box
from env import DominionEnv
from callbacks import DomCallbacks
from provincial_vs_provincial import buy_menu_one as preset_menu
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray import tune
from ray.tune import run_experiments, register_env
from ray.rllib.models import ModelCatalog
from domrl.agents.provincial_agent import ProvincialAgent
from domrl.engine.supply import BasicPiles
from domrl.engine.cards.base import BaseKingdom
import argparse

torch, nn = try_import_torch()


# for available actions check out:
# https://github.com/ray-project/ray/blob/739f6539836610e3fbaadd3cf9ad7fb9ae1d79f9/rllib/examples/models/parametric_actions_model.py
class DomrayModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        true_obs_space = Box(low=0.0, high=55.0, shape=(100,), dtype=np.float32)
        self.torch_sub_model = TorchFC(true_obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        
        action_logits, _ = self.torch_sub_model({
            "obs": input_dict["obs"]["state"] # NOTE: maybe need to add .float()?
        })
        
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        #import pdb; pdb.set_trace()
        return action_logits + inf_mask, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])



allowed_cards = ['Cellar', 'Chapel', 'Moat', 'Village', \
    'Bureaucrat', 'Gardens', 'Militia', \
    'Smithy', 'Throne Room', 'Council Room', 'Festival', \
    'Laboratory', 'Library', 'Market', 'Mine', 'Witch'] 
    # Harbinger, Merchant, Vassal, Poacher, Bandit, 
    # Sentry, Artisan, Workshop, Remodel, Moneylender

preset_cards = ['Village', 'Bureaucrat', 'Smithy', 'Witch', 'Militia', 'Moat', 'Library', 'Market', 'Mine', 'Council Room']

preset_supply = {k:v for k,v in BasicPiles.items()}
for card in preset_cards:
    preset_supply[card] = BaseKingdom[card]


env_config = {
    'agents': [ProvincialAgent, ProvincialAgent],
    'buy_menus': [[],[]],
    'players': None,
    'preset_supply': preset_supply,
    'kingdoms': [BaseKingdom],
    'verbose': False
}

eval_config = {
    'env_config': {
        'agents': [ProvincialAgent, ProvincialAgent],
        'buy_menus': [[], preset_menu],
        'players': None,
        'preset_supply': preset_supply,
        'kingdoms': [BaseKingdom],
        'verbose': True
    }
}


num_episodes_per_scenario = 20
eval_metrics = []
def provincial_eval(trainer, eval_workers):
    """Evaluates the performance of the domray model by playing it against
    Provincial using preset buy menus.

    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.

    Returns:
        metrics (dict): evaluation metrics dict
    """
    global eval_metrics

    for i in range(num_episodes_per_scenario):
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])
        #for worker in eval_workers.remote_workers():
        #    worker.foreach_env.remote(lambda env: env.debug())

    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=600)

    metrics = summarize_episodes(episodes)
    eval_metrics.append(metrics)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    ray.init()

    register_env("dominion", lambda config: DominionEnv(config))
    ModelCatalog.register_custom_model("domraymodel", DomrayModel)
    
    config = {
        "env": DominionEnv,
        "env_config": env_config,
        "num_gpus": 1,
        "train_batch_size": 200,
        "model": {
            "custom_model": "domraymodel",
            "fcnet_hiddens": [256, 256, 34], #TODO: 34 is the action space size, refactor
            "vf_share_layers": True,
        },
        "callbacks": DomCallbacks,

        # Evaluation settings
        "evaluation_interval": 100,
        "evaluation_num_episodes": 5,
        "evaluation_config": eval_config,
        "evaluation_num_workers": 1,
        "custom_eval_function": provincial_eval,

        "_use_trajectory_view_api": False,
        "framework": "torch",
        "dueling": False,
        "hiddens": [],
        "double_q": False,
        "num_workers": 5,
        "train_batch_size": 32,

    }

    stop = {
        "training_iteration": args.num_iters
    }

    results = tune.run("DQN", config=config, stop=stop)
    lr = results.trials[0].last_result
    import pdb; pdb.set_trace()

    pass
