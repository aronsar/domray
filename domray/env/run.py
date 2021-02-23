import ray
from env import DominionEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray import tune
from ray.tune import run_experiments, register_env
from ray.rllib.models import ModelCatalog
from domrl.agents.provincial_agent import ProvincialAgent
from domrl.engine.supply import BasicPiles
from domrl.engine.cards.base import BaseKingdom
import argparse

torch, nn = try_import_torch()

class DomrayModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

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
    'agents': [ProvincialAgent(), ProvincialAgent()],
    'players': None,
    'preset_supply': preset_supply,
    'kingdoms': [BaseKingdom],
    'verbose': False
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    ray.init()

    register_env("dominion", lambda config: DominionEnv(config))
    ModelCatalog.register_custom_model("domraymodel", DomrayModel)
    
    config = {
        "env": DominionEnv,
        "env_config": env_config,
        "num_gpus": 1,
        "model": {
            "custom_model": "domraymodel",
            "vf_share_layers": True,
        },
        "framework": "torch",
    }

    stop = {
        "training_iteration": args.num_iters
    }

    results = tune.run("DQN", config=config, stop=stop)


    '''
    run_experiments({
        "dqn_dom": {
            "run": "DQN",
            "env": "dominion",
            "stop": {
                "training_iteration": args.num_iters,
            },
            "config": {
                "num_workers": args.num_workers,
                #"observation_filter": "NoFilter",  # breaks the action mask
                #"vf_share_layers": True,  # don't create a separate value model
                "env_config": env_config,
                "model": {
                    "custom_model": "domraymodel",
                },
            },
        },
     })
     '''
