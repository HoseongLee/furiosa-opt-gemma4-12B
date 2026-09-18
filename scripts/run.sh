#!/bin/bash
set -euo pipefail

cargo furiosa-opt run --release --bin gemma4 -- "$@"
