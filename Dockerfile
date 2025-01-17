# Use a minimal base image
FROM debian:bookworm-slim

# Copy the installation script into the container
COPY DockerInstall.sh /DockerInstall.sh
RUN chmod +x /DockerInstall.sh

# Use BuildKit Secrets to pass GitHub credentials to the script
RUN --mount=type=secret,id=gh_username \
    --mount=type=secret,id=gh_pat \
    bash /DockerInstall.sh $(cat /run/secrets/gh_username) $(cat /run/secrets/gh_pat)

# Set final working directory
WORKDIR /app

# Default shell to bash
SHELL ["/bin/bash", "-c"]

# Set entry point
CMD ["/bin/bash"]

