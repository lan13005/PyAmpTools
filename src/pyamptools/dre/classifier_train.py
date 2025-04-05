import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from flax import nnx

class DensityRatioEstimator(nnx.Module):
    
    def __init__(self, dims, dropout_rate=0.2, rngs=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layers = []
        for din, dout in zip(dims[:-1], dims[1:]):
            self.layers.append(nnx.Linear(din, dout, rngs=rngs))
        self.dropout = nnx.Dropout(rate=self.dropout_rate, rngs=rngs)
        self.output_layer = nnx.Linear(dims[-1], 1, rngs=rngs)

    def __call__(self, x):
        for layer in self.layers:
            x = nnx.relu(self.dropout(layer(x)))
        return self.output_layer(x)

