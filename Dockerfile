# Use a minimal base image for alma linux (matching JLab)
FROM almalinux:latest

# Copy the installation script into the container
COPY DockerInstall.sh /DockerInstall.sh

WORKDIR /app

########################################################################################################################
# (BUILDING ON MACOS) has certificates in a different location than linux, leads to SSL errors!
#    DockerBuild.sh exports certificates to root-certificates.crt which copies into container
RUN mkdir -p /etc/pki/ca-trust/source/anchors/
COPY root-certificates.crt /etc/pki/ca-trust/source/anchors/
########################################################################################################################

RUN chmod +x /DockerInstall.sh

# Use BuildKit Secrets to pass GitHub credentials to the script
RUN --mount=type=secret,id=gh_username \
    --mount=type=secret,id=gh_pat \
    bash /DockerInstall.sh $(cat /run/secrets/gh_username) $(cat /run/secrets/gh_pat)

# Default shell to bash
SHELL ["/bin/bash", "-c"]

# Set entry point
CMD ["/bin/bash"]

