#!/usr/bin/env bash
# Starts the "sandboxed edge" configuration for the 3-way deployment

set -euo pipefail

IMAGE_NAME="$(basename "$(pwd)")-edge_service"
CONTAINER_NAME="vcse_edge_sandboxed"
VOLUME_NAME="$(basename "$(pwd)")_artifact-store"
NETWORK_NAME="$(basename "$(pwd)")_vcse-net"
HOST_PORT=8002

echo "   If this fails with 'no such image/volume/network', check the exact"
echo "    names with: docker images | docker volume ls | docker network ls"
echo "    (compose prefixes names with your project folder name, which this"
echo "    script guesses from the current directory  override the"
echo "    IMAGE_NAME / VOLUME_NAME / NETWORK_NAME variables above if needed)"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  --cpus="0.5" \
  --memory="256m" \
  --network "${NETWORK_NAME}" \
  -p "${HOST_PORT}:8000" \
  -v "${VOLUME_NAME}:/data:ro" \
  -e NODE_ROLE=edge-sandboxed \
  "${IMAGE_NAME}"

# Strict, no-network alternative (uncomment to use; requires benchmark_client
# to call this container via `docker exec` instead of HTTP, since there is no
# route in or out with --network none):
#
# docker run -d \
#   --name "${CONTAINER_NAME}" \
#   --cpus="0.5" \
#   --memory="256m" \
#   --network none \
#   -v "${VOLUME_NAME}:/data:ro" \
#   -e NODE_ROLE=edge-sandboxed \
#   "${IMAGE_NAME}"

echo " OK, Sandboxed edge container started as ${CONTAINER_NAME} on host port ${HOST_PORT}."
echo "     Verify limits with: docker inspect ${CONTAINER_NAME} | grep -A3 -i '\"Memory\"\\|NanoCpus'"
