#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${APP_DIR}/static/vendor"
mkdir -p "${VENDOR_DIR}"
cp "${APP_DIR}/node_modules/three/build/three.module.js" "${VENDOR_DIR}/three.module.js"
cp "${APP_DIR}/node_modules/three/examples/jsm/controls/OrbitControls.js" "${VENDOR_DIR}/OrbitControls.js"
cp "${APP_DIR}/node_modules/lucide/dist/umd/lucide.js" "${VENDOR_DIR}/lucide.js"
printf 'Vendored Three.js, OrbitControls, and Lucide into %s\n' "${VENDOR_DIR}"
