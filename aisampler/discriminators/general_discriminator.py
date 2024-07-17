import jax.numpy as jnp
import flax.linen as nn
from aisampler.kernels import create_henon_flow

class GeneralDiscriminator(nn.Module):
    L: nn.Module
    GD: nn.Module
    d: int

    def setup(self) -> None:

        self.R = jnp.concatenate(
            jnp.array([[1.0 for _ in range(self.d)], [-1.0 for _ in range(self.d)]])
        )

    def __call__(self, x):
        trans_x =  self.R * self.L(x)
        inp = jnp.concatenate(trans_x, x)
        # Choose the first output. [0] should be equal to -1 * [1].
        return self.GD(inp)[0]

# Linear Layer of the form
# | A B |
# | B A |
class EquivariantLinear(nn.Module):
    num_input: int
    num_output: int
    def setup(self) -> None:
        assert self.num_input % 2 == 0
        assert self.num_output % 2 == 0
        self.A = self.param('A', nn.initializers.lecun_normal(), (self.num_output / 2, self.num_input / 2))
        self.B = self.param('B', nn.initializers.lecun_normal(), (self.num_output / 2, self.num_input / 2))

    def __call__(self, x):
        # Construct the full weight matrix
        top = jnp.concatenate([self.A, self.B], axis=1)
        bottom = jnp.concatenate([self.B, self.A], axis=1)
        weights = jnp.concatenate([top, bottom], axis=0)
        
        return jnp.dot(x, weights)

def create_general_discriminator(
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
        GD=nn.Sequential(
            [EquivariantLinear(num_input=d, num_output=num_hidden_d), activation]
            + [
                EquivariantLinear(num_input=num_hidden_d, num_output = num_hidden_d),
                activation
            ]
            *
            (num_layers_d - 1)
            + [EquivariantLinear(num_input=num_hidden_d, num_output=2)]
        ),
        d = d
    )






