# iftpwa - Tips

## Analysis Tips

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