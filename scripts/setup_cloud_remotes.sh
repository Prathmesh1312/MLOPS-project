#!/usr/bin/env bash
# ==============================================================================
# Configure DVC Remotes:
# 1. `local_cloud_cache` (Default for instant offline local execution)
# 2. `cloudflare_r2` (Zero-egress S3-compatible Cloudflare R2 remote for cloud CI/CD/CT)
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DVC_BIN="${PROJECT_DIR}/.venv/bin/dvc"
if [ ! -x "$DVC_BIN" ]; then
    DVC_BIN="dvc"
fi

if [ ! -d ".dvc" ]; then
    "$DVC_BIN" init --no-scm || "$DVC_BIN" init -f
fi

mkdir -p "${PROJECT_DIR}/.dvc_local_storage"

# 1. Local remote for 100% self-contained local machine execution
"$DVC_BIN" remote add -d -f local_storage "${PROJECT_DIR}/.dvc_local_storage"

# 2. Cloudflare R2 zero-egress S3-compatible object storage remote
R2_ACCOUNT_ID="${CF_ACCOUNT_ID:-<CLOUDFLARE_ACCOUNT_ID>}"
R2_BUCKET="${R2_BUCKET_NAME:-mlops-evaporation-artifacts}"
"$DVC_BIN" remote add -f cloudflare_r2 "s3://${R2_BUCKET}/dvc-store"
"$DVC_BIN" remote modify cloudflare_r2 endpointurl "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
"$DVC_BIN" remote modify cloudflare_r2 region "auto"

echo "[OK] DVC remotes configured:"
"$DVC_BIN" remote list
