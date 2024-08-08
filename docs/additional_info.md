# Potential Build Errors

## Failure to pip install mpi4py

If installing `mpi4py` fails due to `error: Cannot link MPI programs` this is a common conda-forge linker issue. Try replacing the built-in linker with the system's and attempt to install `mpi4py` again.

```shell
# If you havent done so already please activate your conda environment first
rm $CONDA_PREFIX/compiler_compat/ld
ln -s /usr/bin/ld $CONDA_PREFIX/compiler_compat/
```
