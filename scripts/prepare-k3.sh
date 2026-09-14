#!/usr/bin/env bash
exec "$(cd "$(dirname "$0")" && pwd)/prepare" k3 "$@"
