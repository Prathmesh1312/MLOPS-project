#!/usr/bin/env bash
# ==============================================================================
# Zero-Downtime Blue-Green Container Deployment Script (AWS EC2 & Local Docker)
# ==============================================================================
set -euo pipefail

IMAGE_TAG="${1:-ghcr.io/shashwat-upadhyay/ml-ci-cd-ct-pipeline:latest}"
ACTIVE_SLOT_FILE="/tmp/mlops_active_slot"

if [ -f "$ACTIVE_SLOT_FILE" ]; then
    CURRENT_SLOT="$(cat "$ACTIVE_SLOT_FILE")"
else
    CURRENT_SLOT="blue"
fi

if [ "$CURRENT_SLOT" = "blue" ]; then
    NEXT_SLOT="green"
    NEXT_PORT=8002
    OLD_CONTAINER="mlops-api-blue"
    NEW_CONTAINER="mlops-api-green"
else
    NEXT_SLOT="blue"
    NEXT_PORT=8001
    OLD_CONTAINER="mlops-api-green"
    NEW_CONTAINER="mlops-api-blue"
fi

echo "=================================================================="
echo "[Cloud CD] Starting Zero-Downtime Blue-Green Deployment"
echo "           Current Slot : ${CURRENT_SLOT}"
echo "           Target Slot  : ${NEXT_SLOT} (Port ${NEXT_PORT})"
echo "           Image        : ${IMAGE_TAG}"
echo "=================================================================="

docker rm -f "${NEW_CONTAINER}" >/dev/null 2>&1 || true

docker run -d \
    --name "${NEW_CONTAINER}" \
    -e DEPLOYMENT_SLOT="${NEXT_SLOT}" \
    -p "${NEXT_PORT}:8000" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/logs:/app/logs" \
    "${IMAGE_TAG}"

echo "[Cloud CD] Waiting for Readiness Probe on ${NEW_CONTAINER} (http://localhost:${NEXT_PORT}/health/ready)..."
READY=0
for i in $(seq 1 20); do
    if curl -fsS "http://localhost:${NEXT_PORT}/health/ready" >/dev/null 2>&1; then
        READY=1
        echo "[Cloud CD] Readiness probe passed on attempt ${i}!"
        break
    fi
    sleep 2
done

if [ "$READY" -ne 1 ]; then
    echo "[ERROR] New container (${NEW_CONTAINER}) failed readiness probe! Rolling back..."
    docker rm -f "${NEW_CONTAINER}" >/dev/null 2>&1 || true
    exit 1
fi

echo "${NEXT_SLOT}" > "$ACTIVE_SLOT_FILE"

# If Nginx load balancer container is running, reload its upstream configuration
if docker ps --format '{{.Names}}' | grep -q "^mlops-nginx-lb$"; then
    docker exec mlops-nginx-lb nginx -s reload
    echo "[Cloud CD] Nginx Reverse Proxy upstream switched to ${NEXT_SLOT} with zero dropped requests."
fi

echo "[Cloud CD] Draining previous container (${OLD_CONTAINER})..."
docker stop -t 10 "${OLD_CONTAINER}" >/dev/null 2>&1 || true

echo "[SUCCESS] Zero-Downtime Deployment Complete! Active Slot: ${NEXT_SLOT}"
