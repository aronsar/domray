from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
import ray

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

    for i in range(num_episodes_per_scenario):
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])
        #for worker in eval_workers.remote_workers():
        #    worker.foreach_env.remote(lambda env: env.debug())

    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=600)

    metrics = summarize_episodes(episodes)
    import pdb; pdb.set_trace()

    return metrics

