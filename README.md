# Installation

Set some required environment variables

```shell
source set_environment.sh
```

To install AmpTools as a submodule:

```shell
git clone https://github.com/lan13005/PyAmpTools --recurse-submodules
cd external/MakeAmpTools/
make # distributes modified Makefiles and makes AmpTools
```

Environment setup. `conda` can be very slow to resolve dependencies. I recommend [Mamba](https://github.com/conda-forge/miniforge#mambaforge)

```shell
conda env create                    # Creates environment specified by environment.yml and pyproject.toml
conda activate PyAmpTools           # activate the environment
mamba install -c conda-forge root   # install root
```
