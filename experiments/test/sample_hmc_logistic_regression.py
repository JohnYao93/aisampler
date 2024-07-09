from pathlib import Path
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

import aisampler.logistic_regression as logistic_regression
from aisampler.logistic_regression import (
    plot_histograms2d_logistic_regression,
    plot_histograms_logistic_regression,
    plot_logistic_regression_samples,
)
from aisampler.sampling import hmc, effective_sample_size

_TASK_FILE = config_flags.DEFINE_config_file(
    "task", default="experiments/config/config_sample_hmc_logistic_regression.py"
)


def load_cfgs(
    _TASK_FILE,
):
    cfg = _TASK_FILE.value

    return cfg


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = getattr(logistic_regression, cfg.dataset_name)(
        batch_size=cfg.num_parallel_chains,
        mode="train",
    )

    grad_potential_fn = density.get_grad_energy_fn()

    logging.info(f"Sampling from {cfg.dataset_name} density...")

    samples, ar = hmc(
        density=density,
        grad_potential_fn=grad_potential_fn,
        cov_p=jnp.eye(density.dim) * 1.0,
        d=density.dim,
        parallel_chains=cfg.num_parallel_chains,
        num_steps=cfg.num_steps,
        step_size=cfg.step_size,
        n=cfg.num_iterations,
        burn_in=cfg.burn_in,
        initial_std=0.1,
        rng=jax.random.PRNGKey(cfg.seed),
        vstack=False,
    )

    logging.info(f"Sampling done. Acceptance rate: {ar}")

    # Compute ESS

    ess = effective_sample_size(
        samples[:, :, : density.dim],
        np.array(density.mean()),
        np.array(density.std()),
    )

    for i in range(density.dim):
        logging.info(f"ESS w_{i}: {ess[i]}")

    # Plot

    samples = np.vstack(np.transpose(np.array(samples), (1, 0, 2)))

    np.set_printoptions(linewidth=200, precision=4, suppress=True)
    print(
        "MEAN:    ",
        np.array2string(np.mean(samples, axis=0)[: density.dim], separator=","),
    )
    print("GT MEAN: ", np.array2string(density.mean(), separator=","))
    print(
        "STD:    ",
        np.array2string(np.std(samples, axis=0)[: density.dim], separator=","),
    )
    print("GT STD: ", np.array2string(density.std(), separator=","))

    plot_logistic_regression_samples(
        samples,
        density,
    )

    plot_histograms_logistic_regression(
        samples,
    )

    plot_histograms2d_logistic_regression(
        samples,
    )

    test_density = getattr(logistic_regression, cfg.dataset_name)(
        batch_size=samples.shape[0],
        mode="test",
    )

    v = samples[:, : density.dim]
    score = np.zeros(test_density.data[0].shape[0])
    for i, (x, y) in enumerate(zip(test_density.data[0], test_density.labels[0])):
        score[i] = jax.scipy.special.logsumexp(
            -test_density.sigmoid(v, x, y, test_density.x_dim, test_density.y_dim)
        ) - jnp.log(v.shape[0])

    logging.info(f"Average predictive posterior: {score.mean()}")


if __name__ == "__main__":
    app.run(main)
