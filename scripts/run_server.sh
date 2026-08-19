#!/bin/bash
set -euo pipefail

cargo furiosa-opt test --release
cargo furiosa-opt run --release --bin server
