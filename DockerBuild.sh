#!/bin/bash

# Enable BuildKit for multi-architecture support
export DOCKER_BUILDKIT=1

# Check if GitHub credentials are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <GH_USERNAME> <GH_PAT>"
fi

# CLI GitHub username and personal access token for cloning over HTTPS
GH_USERNAME=$1
GH_PAT=$2

########################################################################################################################
# (BUILDING ON MACOS) Export macOS system certificates to root-certificates.crt
security export -t certs -f pemseq -k /System/Library/Keychains/SystemRootCertificates.keychain > root-certificates.crt
security export -t certs -f pemseq -k /Library/Keychains/System.keychain >> root-certificates.crt
########################################################################################################################

# Run Docker Buildx with secrets, some platform options linux/amd64 or linux/arm64
docker buildx build \
    --platform linux/amd64 \
    --no-cache \
    --secret id=gh_username,src=<(echo -n "$GH_USERNAME") \
    --secret id=gh_pat,src=<(echo -n "$GH_PAT") \
    --output type=docker,dest=pyamptools.tar .

# Can be done on the cluster
# apptainer build pyamptools.sif docker-archive://pyamptools.tar