from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
import ray
#from domrl.engine.supply import BasicPiles
#from domrl.engine.cards.base import BaseKingdom
#from provinddcial_vs_provincial import buy_menu_one as preset_menu

num_episodes_per_scenario = 6
preset_cards = ['Village', 'Bureaucrat', 'Smithy', 'Witch', 'Militia', 'Moat', 'Library', 'Market', 'Mine', 'Council Room']

def provincial_eval(trainer, eval_workers):
    """Evaluates the performance of the domray model by playing it against
    Provincial using preset buy menus.

    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.

    Returns:
        metrics (dict): evaluation metrics dict
    """

    #eval_scenarios = [(preset_cards, preset_menu)]

    # iterate through different card set/buy menu combinations
    #for (card_list, buy_menu), worker in zip(eval_scenarios, eval_workers.remote_workers()):
        #preset_supply = {k:v for k,v in BasicPiles.items()}
        #for card in card_list:
            #preset_supply[card] = BaseKingdom[card]

        #env_config = {
        #    'agents': [ProvincialAgent([]), ProvincialAgent(buy_menu)],
        #    'players': None,
        #    'preset_supply': preset_supply,
        #    'kingdoms': [BaseKingdom],
        #    'verbose': True
        #}

        #worker.foreach_env.remote(lambda env: env.update_config(env_config))

    
    for i in range(num_episodes_per_scenario):
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])
        for worker in eval_workers.remote_workers():
            worker.foreach_env.remote(lambda env: env.debug())

    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=600)

    metrics = summarize_episodes(episodes)

    return metrics

