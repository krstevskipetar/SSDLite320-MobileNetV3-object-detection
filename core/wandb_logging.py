import wandb

import core.plotting


def log_to_wandb(project_name: str, config: dict, metrics: dict, figures: list, checkpoint_path: str, step=None, plot=False):
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        # track hyperparameters and run metadata
        config=config,
        id=config['id']
    )
    config['id'] = wandb.run.id
    wandb.log(metrics)

    for idx, f in enumerate(figures):
        wandb.log({f"batch_predictions_{idx}": f}, step=step)

    if plot:
        metrics_figure = core.plotting.plot_metrics(metrics)
        wandb.log({"metrics_plot": metrics_figure})
    wandb.log_artifact(checkpoint_path)
