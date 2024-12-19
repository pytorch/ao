#!/bin/bash
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --fix      Fix format (default)"
  echo "  --check    Check for formatting issues"
  exit 1
}

# Parse the command-line options
# TEMP=$(getopt -o '' --long fix,check -- "$@")
# eval set -- "$TEMP"
FIX=true
case "$1" in
    --fix)
        FIX=true
        break
        ;;
    --check)
        FIX=false
        break
        ;;
    "")
        FIX=true
        break
        ;;
    *)
        usage
        break
        ;;
esac

# Execute the desired action based on the value of FIX_ISSUES
if $FIX; then
    ruff check . --fix
    # --isolated is used to skip the allowlist at all so this applies to all files
    # please be careful when using this large changes means everyone needs to rebase
    ruff check --isolated --select F821,F823,W191 --fix
    ruff check --select F,I --fix
    ruff format .
else
    ruff check .
    # --isolated is used to skip the allowlist at all so this applies to all files
    # please be careful when using this large changes means everyone needs to rebase
    ruff check --isolated --select F821,F823,W191
    ruff check --select F,I
    ruff format --check || {
        echo "Ruff check failed, please try again after running scripts/run_ruff.sh"
        exit 1
    }
fi
