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
        trans_x =  self.R * self.L(x)
        inp = jnp.concatenate([trans_x, x], axis=1)
        # Choose the first output. [0] should be equal to -1 * [1].
        return self.D(inp)[:,0]
    
    def __to_onnx(self):
        dummy_input = 1

# Linear Layer of the form
# | A B |
# | B A |
class EquivariantLinear(nn.Module):
    num_output: int
    
    @nn.compact
    def __call__(self, x):
        input_features = x.shape[-1]
        assert input_features % 2 == 0
        assert self.num_output % 2 == 0

        A = self.param("A", nn.initializers.lecun_normal(), (input_features // 2, self.num_output // 2))
        B = self.param("B", nn.initializers.lecun_normal(), (input_features // 2, self.num_output // 2))
        # Construct the full weight matrix
        top = jnp.concatenate([A, B], axis=1)
        bottom = jnp.concatenate([B, A], axis=1)
        weights = jnp.concatenate([top, bottom], axis=0)

        return x @ weights

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
        D=nn.Sequential(
            [EquivariantLinear(num_output=num_hidden_d), activation] + 
            [
                EquivariantLinear(num_output=num_hidden_d),
                activation
            ]
            *
            (num_layers_d - 1)
            + [EquivariantLinear(num_output=2)]
        ),
        d = d
    )






