#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f static/vendor/three.module.js || ! -f static/vendor/lucide.js ]]; then
  if [[ ! -d node_modules ]]; then
    npm ci
  fi
  ./vendor_assets.sh
fi

exec "${APP_PYTHON:-python}" server.py "$@"
