#!/bin/bash

set -e  # Exit on error
set -o pipefail  # Fail on first error in pipelines

# GitHub credentials (passed from Docker BuildKit secrets)
GH_USERNAME=$1
GH_PAT=$2

echo "Using GitHub credentials for cloning private repositories..."

# Overwrite /bin/sh with bash (some Debian images use dash by default)
ln -sf /bin/bash /bin/sh

########################################################################################################################
# (OPTION 1: DEBIAN) Update package list and install dependencies
# apt update && apt install -y --no-install-recommends \
#     git curl unzip vim build-essential python3-dev python3-pip cmake wget \
#     libopenmpi-dev unzip libboost-dev libboost-math-dev libsqlite3-dev \
#     binutils cmake dpkg-dev g++ gcc libssl-dev libx11-dev \
#     libxext-dev libxft-dev libxpm-dev python3 libtbb-dev libvdt-dev libgif-dev \
#     gfortran libpcre3-dev libglu1-mesa-dev libglew-dev libftgl-dev \
#     libfftw3-dev libcfitsio-dev libgraphviz-dev libavahi-compat-libdnssd-dev \
#     libldap2-dev python3-numpy libxml2-dev libkrb5-dev \
#     libgsl-dev qtwebengine5-dev nlohmann-json3-dev libmariadb-dev \
#     libgl2ps-dev liblzma-dev libxxhash-dev liblz4-dev libzstd-dev \
#     ca-certificates && update-ca-certificates && \
#     rm -rf /var/lib/apt/lists/*  # Clean package lists to reduce image size
########################################################################################################################
# (OPTION 2: ALMA LINUX) Update package list and install dependencies
echo "sslverify=false" >> /etc/dnf/dnf.conf # temporary disable ssl verification to install ca-certificates
dnf install -y ca-certificates p11-kit
sed -i '/sslverify=false/d' /etc/dnf/dnf.conf # reenable ssl verification
update-ca-trust extract && update-ca-trust && dnf clean all && dnf makecache # force update of dnf cache with new certificates
dnf install -y epel-release # for installing xrootd
dnf config-manager --set-enabled crb # for installing xrootd
dnf install -y --allowerasing \
    git curl unzip vim make automake gcc gcc-c++ kernel-devel patch environment-modules \
    python3-devel python3-pip cmake wget \
    openmpi openmpi-devel boost-devel sqlite-devel \
    binutils libX11-devel libXpm-devel libXft-devel libXext-devel python openssl-devel \
    xrootd-client-devel xrootd-libs-devel \
    gcc-gfortran pcre-devel \
    mesa-libGL-devel mesa-libGLU-devel glew-devel ftgl-devel mysql-devel \
    fftw-devel cfitsio-devel graphviz-devel libuuid-devel \
    avahi-compat-libdns_sd-devel openldap-devel python-devel python3-numpy \
    libxml2-devel gsl-devel readline-devel qt5-qtwebengine-devel \
    R-devel R-Rcpp-devel R-RInside-devel && \
    dnf clean all  # Clean package lists to reduce image size
source /etc/profile.d/modules.sh
########################################################################################################################

# Install Conda (Mamba) from Miniforge
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p "${HOME}/conda"
rm -f Miniforge3.sh
source "${HOME}/conda/etc/profile.d/conda.sh"
conda activate

# Clone PyAmpTools using BuildKit Secrets
git clone https://${GH_USERNAME}:${GH_PAT}@github.com/lan13005/PyAmpTools.git --recurse-submodules
cd PyAmpTools
conda create --name pyamptools python=3.9 -y
conda activate pyamptools
pip install -e .[dev]
conda install -c conda-forge tbb -y
source set_environment.sh
# update linker for mpi4py
rm -f $CONDA_PREFIX/compiler_compat/ld
ln -s /usr/bin/ld $CONDA_PREFIX/compiler_compat/ld
module unload mpi && module load mpi
pip install mpi4py
cd ..

# Clone IFTPWA using BuildKit Secrets
git clone https://${GH_USERNAME}:${GH_PAT}@github.com/fmkroci/iftpwa.git
cd iftpwa
git checkout hyperopt
pip install -e .[dev]
cd ..

# nifty8 actually requires newer versions of python > 3.9 and therefore updated numpy and matplotlib
#   FUTURE: update iftpwa to use newer python otherwise we wont have access to later versions of numpy and matplotlib
NIFTY_SITE="$CONDA_PREFIX/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nifty8/"
sed -i 's/np.asfarray(x)/np.asarray(x, dtype=float)/g' $NIFTY_SITE/utilities.py
sed -i '' '/^def _register_cmaps():/,/^def /s|import matplotlib\.pyplot as plt|from matplotlib import colormaps as cm|' $NIFTY_SITE/plot.py
sed -i 's/plt.register_cmap/cm.register/g' $NIFTY_SITE/plot.py

# Install and setup fzf using BuildKit Secrets
git clone --depth 1 https://${GH_USERNAME}:${GH_PAT}@github.com/junegunn/fzf.git /app/fzf
/app/fzf/install --all

# There is a known conflict between AmpTools' GPU usage and RooFit/TMVA which comes with the conda-forge binaries of ROOT. 
# Currently, ROOT has to be built from source with roofit and tmva off. 
# A build script is included to download ROOT from source with the appropriate cmake flags to achieve this. 
# [ROOT](https://root.cern/install/) >v6.26 is a required dependency (build steps are shown below) requiring at least some version of `gcc` (9.3.0 works). 
cd PyAmpTools/external/root
source build_root.sh
cd ../..
# source set_environment.sh  # Reload environment variables

# # Build external dependencies with MPI support
# cd external
# make mpi

# Add aliases to .bashrc
echo 'alias ls="ls --color=auto"' >> /etc/bash.bashrc && \
echo 'alias ll="ls -l --color=auto"' >> /etc/bash.bashrc && \
echo 'alias st="git status"' >> /etc/bash.bashrc && \
echo 'alias log="git log --graph --abbrev-commit --decorate --format=format:'"'"'%C(bold blue)%h%C(reset) - %C(bold cyan)%aD%C(reset) %C(bold green)(%ar)%C(reset)%C(bold yellow)%d%C(reset)%n''          %C(white)%s%C(reset) %C(dim white)- %an%C(reset)'"'"' --all"' >> /etc/bash.bashrc && \
echo 'alias root="root -l"' >> /etc/bash.bashrc && \
echo 'source /etc/profile.d/modules.sh' >> /etc/bash.bashrc && \
echo 'module unload mpi && module load mpi' >> /etc/bash.bashrc && \
echo 'source /root/conda/bin/activate' >> /etc/bash.bashrc && \
echo 'conda activate pyamptools' >> /etc/bash.bashrc && \
echo 'cd /app/PyAmpTools/ && source set_environment.sh && cd /app/PyAmpTools/' >> /etc/bash.bashrc && \
echo 'if [ -f /etc/bash.bashrc ]; then source /etc/bash.bashrc; fi' >> ~/.bashrc
