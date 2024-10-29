from unittest.mock import patch
from absl import app
from ml_collections import config_flags
from pathlib import Path
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax

try:
    import wandb
except ImportError:
    wandb = None

import aisampler.toy_densities as densities
import aisampler.sampling as sampling
from aisampler.trainers import Trainer


_TASK_FILE = config_flags.DEFINE_config_file(
    "task", default="experiments/config/config_train_toy_density.py"
)


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    Path(cfg.figure_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if cfg.wandb.use:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg)

    density = getattr(densities, cfg.target_density_name)

    if not cfg.train.quiet:
        densities.plot_hamiltonian_density(density)

    trainer = Trainer(
        cfg=cfg,
        density=density,
    )

    trainer.train_model()

    samples, ar = trainer.sample_fn(
        trainer.L_state.params,
        jax.random.PRNGKey(42),
    )

    if not cfg.train.quiet:  
        sampling.plot_samples_with_density(
            samples=samples, target_density=density, ar=ar, name=None
        )


if __name__ == "__main__":
    app.run(main)
