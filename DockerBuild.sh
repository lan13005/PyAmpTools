#!/bin/bash

# Enable BuildKit for multi-architecture support
export DOCKER_BUILDKIT=1

# CLI GitHub username and personal access token for cloning over HTTPS
GH_USERNAME=$GH_USERNAME
GH_PAT=$GH_PAT

if [ -z "$GH_USERNAME" ] || [ -z "$GH_PAT" ]; then
    echo "GH_USERNAME and GH_PAT must be set"
    exit 1
fi

########################################################################################################################
# (BUILDING ON MACOS) Export macOS system certificates to root-certificates.crt
security export -t certs -f pemseq -k /System/Library/Keychains/SystemRootCertificates.keychain > root-certificates.crt
security export -t certs -f pemseq -k /Library/Keychains/System.keychain >> root-certificates.crt
########################################################################################################################

# Create a new buildx builder instance
docker buildx create --name pyamptools_builder --use

# Run Docker Buildx with secrets, some platform options linux/amd64 or linux/arm64
docker buildx build \
    --platform linux/amd64 \
    --no-cache \
    --secret id=gh_username,src=<(echo -n "$GH_USERNAME") \
    --secret id=gh_pat,src=<(echo -n "$GH_PAT") \
    --output type=docker,dest=pyamptools.tar .

# To be done manually on the cluster
# apptainer build pyamptools.sif docker-archive://pyamptools.tar # build singularity image file from archived tar file
# apptainer exec --contain --bind <working_dir> --bind <data_dir> --env BASH_ENV=/dev/null pyamptools.sif bash --noprofile --norc