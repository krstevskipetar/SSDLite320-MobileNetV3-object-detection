import plotting
import wandb


def log_to_wandb(project_name: str, config: dict, metrics: dict, figures: list, checkpoint_path: str):
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        # track hyperparameters and run metadata
        config=config
    )
    wandb.log(metrics)

    for idx, f in enumerate(figures):
        wandb.log({f"batch_predictions_{idx}": f})

    metrics_figure = plotting.plot_metrics(metrics)
    wandb.log({"metrics_plot": metrics_figure})
    wandb.log_artifact(checkpoint_path)
