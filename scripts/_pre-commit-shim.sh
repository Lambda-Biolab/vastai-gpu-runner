#!/usr/bin/env bash
# Pre-commit hook wrapper that adds ~/.local/bin to PATH so
# uv-installed tools are found. The pre-commit framework runs
# hooks in a subshell that doesn't source the user's shell
# profile, so ~/.local/bin is not on the default PATH.
#
# Usage in .pre-commit-config.yaml:
#   entry: .git/hooks/_pre-commit-shim.sh pyright src/
#   language: script
#
# The shim is created by this script (one-time, on hook install).
# Subsequent runs use it transparently.
export PATH="$HOME/.local/bin:$PATH"
exec "$@"
