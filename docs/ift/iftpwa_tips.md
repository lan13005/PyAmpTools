# iftpwa - Tips

## Analysis Tips

Hyperparameter scans are performed under the hood using `optuna` python package. `PyAmpTools` streamlines this interaction, see the configuration file tutorial. We can hijack this interface to perform scans over any hyperparameter defined in the config file.
0. Currently only `iftpwa` uses this feature but one can imagine doing waveset scans with `amptools` fits (to be added). Use `optuna`'s `BruteForceSampler` to perform scans over a list of waveset strings.
1. If we wanted to do random fits with `iftpwa`, one can use `optuna`s `BruteForceSampler` to scan over seeds.
2. We can additionally hijack this system to extract our systematic uncertainties. `NIFTy` researchers estimates systematic uncertainties by randomly moving the prior mean (of all prior distributions) around the prior std, performs a fit, then aggregates the shifts. 

## Developing Tips
- All random variables start normally distributed, when constructing terms/factors in the model it is often needed to transform to other distributions which can be done through using the cumulative distribution functions and quantile functions
- `ift.from_random` is often used to probe what the forward model is doing
- **Example:** code for drawing from a half-normal distribution

```python
from jax.scipy.special import erfinv
from jax.scipy.stats.norm import cdf as normal_cdf
import nifty8 as ift
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

scale = 10 # arbitary standard deviation

domain = ift.RGSpace(shape=(1 ,), distances=(1.0,), harmonic=False)

def fun(x):
    uniform_rv = normal_cdf(x)
    # see https://en.wikipedia.org/wiki/Normal_distribution for Quantile function
    #   which uses erfinv(2p-1) where p is uniform. Makes result half-normal instead of normal
    return scale * jnp.sqrt(2.0) * erfinv(uniform_rv)

fun = ift.JaxOperator(domain=domain, target=domain, func=fun)
draws = np.array([fun(ift.from_random(domain)).val for _ in range(10000)]).flatten()
plt.hist(draws, bins=100)
```