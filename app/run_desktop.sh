#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" && " ${*:-} " != *" --smoke-test "* ]]; then
  echo "No graphical display is available. Use X11 forwarding or a remote desktop." >&2
  exit 2
fi

exec "${APP_PYTHON:-python}" desktop_app.py "$@"
