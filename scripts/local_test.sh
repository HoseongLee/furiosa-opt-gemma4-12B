#!/bin/bash
set -euo pipefail

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CRATE"

FIXTURE="ref/fixtures.safetensors"

for argument in "$@"; do
    case "$argument" in
        -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "local_test.sh: unknown argument $argument" >&2; exit 2 ;;
    esac
done

if [ ! -f "$FIXTURE" ]; then
    echo "local_test.sh: $FIXTURE is missing -- generate it first:" >&2
    echo "    python3 scripts/generate_references.py" >&2
    exit 1
fi

echo "==> running test_kernels"
cargo furiosa-opt test --release
cargo furiosa-opt run --release --bin test_kernels
