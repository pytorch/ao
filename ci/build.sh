#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash ci/build.sh --python=3.10

Options:
  --python=VERSION   Python version suffix used to select pip, e.g. 3.10 -> pip3.10
  --python VERSION   Same as above
  -h, --help         Show this help message
EOF
}

PYVER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python=*)
      PYVER="${1#*=}"
      shift
      ;;
    --python)
      shift
      PYVER="${1:-}"
      shift || true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$PYVER" ]]; then
  echo "Missing required --python argument." >&2
  usage
  exit 1
fi

PIP="pip${PYVER}"
if ! command -v "$PIP" >/dev/null 2>&1; then
  echo "Cannot find ${PIP} on PATH." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p dist
USE_CPP=0 "$PIP" wheel . -w dist --no-build-isolation
