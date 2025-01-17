#!/bin/bash

set -e  # Exit on error
set -o pipefail  # Fail on first error in pipelines

# GitHub credentials (passed from Docker BuildKit secrets)
GH_USERNAME=$1
GH_PAT=$2

echo "Using GitHub credentials for cloning private repositories..."
echo "GitHub Username: $GH_USERNAME"

# Overwrite /bin/sh with bash (some Debian images use dash by default)
ln -sf /bin/bash /bin/sh

# Update package list and install dependencies
apt update && apt install -y --no-install-recommends \
    git curl unzip vim build-essential python3-dev python3-pip cmake wget \
    libopenmpi-dev unzip libboost-dev libboost-math-dev libsqlite3-dev \
    binutils cmake dpkg-dev g++ gcc libssl-dev libx11-dev \
    libxext-dev libxft-dev libxpm-dev python3 libtbb-dev libvdt-dev libgif-dev \
    gfortran libpcre3-dev libglu1-mesa-dev libglew-dev libftgl-dev \
    libfftw3-dev libcfitsio-dev libgraphviz-dev libavahi-compat-libdnssd-dev \
    libldap2-dev python3-numpy libxml2-dev libkrb5-dev \
    libgsl-dev qtwebengine5-dev nlohmann-json3-dev libmariadb-dev \
    libgl2ps-dev liblzma-dev libxxhash-dev liblz4-dev libzstd-dev && \
    rm -rf /var/lib/apt/lists/*  # Clean package lists to reduce image size

# Install Conda (Mamba) from Miniforge
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "${HOME}/conda"
rm Miniforge3.sh
source "${HOME}/conda/etc/profile.d/conda.sh"
conda activate

# Clone PyAmpTools using BuildKit Secrets
git clone https://${GH_USERNAME}:${GH_PAT}@github.com/lan13005/PyAmpTools.git --recurse-submodules
cd PyAmpTools
conda env create
conda activate pyamptools
# autoload set_enviornment.sh on conda activate
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
ln -snfr set_environment.sh $CONDA_PREFIX/etc/conda/activate.d
source set_environment.sh
# update linker for mpi4py
rm $CONDA_PREFIX/compiler_compat/ld
ln -s /usr/bin/ld $CONDA_PREFIX/compiler_compat/con
pip install mpi4py
cd ..

# Clone IFTPWA using BuildKit Secrets
git clone https://${GH_USERNAME}:${GH_PAT}@github.com/fmkroci/iftpwa.git
cd iftpwa
git checkout hyperopt
pip install -e .
cd ..

# Build ROOT
cd PyAmpTools/external/root
source build_root.sh
cd ../..
source set_environment.sh  # Reload environment variables

# Build external dependencies with MPI support
cd PyAmpTools/external
make mpi

