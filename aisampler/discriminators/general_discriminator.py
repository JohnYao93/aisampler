import jax.numpy as jnp
import flax.linen as nn
from aisampler.kernels import create_henon_flow


class GeneralDiscriminator(nn.Module):
    L: nn.Module
    D: nn.Module
    d: int

    def setup(self) -> None:

        self.R = jnp.concatenate(
            jnp.array([[1.0 for _ in range(self.d)], [-1.0 for _ in range(self.d)]])
        )

    def __call__(self, x):
        return 


class EquivariantLinear(nn.Module):
    num_input: int
    num_output: int
    def setup(self) -> None:
        assert self.num_input % 2 == 0
        assert self.num_output % 2 == 0
        self.A = self.param('A', nn.initializers.lecun_normal(), (self.output_features // 2, self.input_features // 2))
        self.B = self.param('B', nn.initializers.lecun_normal(), (self.output_features // 2, self.input_features // 2))

    def __call__(self, x):
        # Construct the full weight matrix
        top = jnp.concatenate([self.A, self.B], axis=1)
        bottom = jnp.concatenate([self.B, self.A], axis=1)
        weights = jnp.concatenate([top, bottom], axis=0)
        
        return jnp.dot(x, weights)

class GD(nn.Module):
    x

def create_general_disciminator(
        num_flow_layers: int,
        num_hidden_flow: int,
        num_layers_flow: int,
        num_layers_d: int,
        num_hidden_d: int,
        activation: str,
        d: int
) -> GeneralDiscriminator:
    activation = getattr(nn, activation)

    return GeneralDiscriminator(
        L=create_henon_flow(
            num_flow_layers=num_flow_layers,
            num_layers=num_layers_flow,
            num_hidden=num_hidden_flow,
            d=d,
        ),

    )






