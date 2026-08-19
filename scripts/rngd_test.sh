#!/bin/bash
set -euo pipefail

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CRATE"

BINARY="target/release/test_kernels"
FIXTURE="ref/fixtures.safetensors"
POLL_SECONDS="${RNGD_POLL_SECONDS:-5}"
TIMEOUT="${RNGD_TIMEOUT:-1800}"

build=1
wait_for_result=1
for argument in "$@"; do
    case "$argument" in
        --no-build) build=0 ;;
        --no-wait) wait_for_result=0 ;;
        -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "rngd_test.sh: unknown argument $argument" >&2; exit 2 ;;
    esac
done

if [ -z "${RNGD_URL:-}" ] && [ -f "$HOME/.bashrc" ]; then
    bashrc_line=$(grep -E '^[[:space:]]*export[[:space:]]+RNGD_URL=' "$HOME/.bashrc" | tail -1 || true)
    if [ -n "$bashrc_line" ]; then
        bashrc_value=${bashrc_line#*=}
        bashrc_value=${bashrc_value%\"}; bashrc_value=${bashrc_value#\"}
        bashrc_value=${bashrc_value%\'}; bashrc_value=${bashrc_value#\'}
        export RNGD_URL="$bashrc_value"
    fi
fi

if [ -z "${RNGD_URL:-}" ]; then
    echo "rngd_test.sh: \$RNGD_URL is not set and ~/.bashrc has no export for it." >&2
    echo "  export RNGD_URL=https://rngd.example.com" >&2
    exit 2
fi

if [ "$build" -eq 1 ]; then
    echo "==> building $BINARY"
    cargo furiosa-opt build --release --bin test_kernels
fi

if [ ! -f "$FIXTURE" ]; then
    echo "rngd_test.sh: $FIXTURE is missing -- generate it first:" >&2
    echo "    python3 scripts/generate_references.py" >&2
    exit 1
fi
for required in "$BINARY" scripts/run_tests.sh; do
    if [ ! -f "$required" ]; then
        echo "rngd_test.sh: $required is missing -- run without --no-build" >&2
        exit 1
    fi
done

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
cp scripts/run_tests.sh "$staging/run_tests.sh"
cp "$BINARY" "$staging/test_runtime"
cp "$FIXTURE" "$staging/fixtures.safetensors"
chmod +x "$staging/run_tests.sh" "$staging/test_runtime"

job_name="${RNGD_JOB_NAME:-rngd_test_$RANDOM}"

echo "==> submitting $job_name ($(du -ch "$staging"/* | tail -1 | cut -f1) total)"
submit_output=$(rngd submit \
    "$staging/run_tests.sh" \
    "$staging/test_runtime" \
    "$staging/fixtures.safetensors" \
    --name "$job_name" \
    --entrypoint run_tests.sh \
    --timeout "$TIMEOUT" 2>&1)
echo "$submit_output"

job=$(printf '%s\n' "$submit_output" | sed -n 's/.*submitted job \([0-9][0-9]*\).*/\1/p' | head -1)
if [ -z "$job" ]; then
    echo "rngd_test.sh: could not find a job id in rngd's output (shown above)" >&2
    exit 1
fi

if [ "$wait_for_result" -eq 0 ]; then
    echo "==> submitted job $job; follow it with: rngd logs $job"
    exit 0
fi

json_field() {
    sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\)\"\{0,1\}.*/\1/p" <<<"$1" | head -1
}

is_terminal() {
    case "$1" in
        succeeded|failed|completed|cancelled|canceled) return 0 ;;
        *) return 1 ;;
    esac
}

echo "==> waiting on job $job (polling every ${POLL_SECONDS}s)"
deadline=$(( SECONDS + TIMEOUT ))
state=""
status_output=""
while [ "$SECONDS" -lt "$deadline" ]; do
    status_output=$(rngd status "$job" 2>&1 || true)
    state=$(json_field "$status_output" status | tr '[:upper:]' '[:lower:]')
    is_terminal "$state" && break
    case "$state" in
        queued|running) ;;
        *) echo "rngd_test.sh: unexpected status '${state:-unreadable}', retrying" >&2 ;;
    esac
    sleep "$POLL_SECONDS"
done

if ! is_terminal "$state"; then
    echo "rngd_test.sh: job $job still '${state:-unknown}' after ${TIMEOUT}s; cancel with: rngd cancel $job" >&2
    exit 1
fi

code=$(json_field "$status_output" exit_code)
echo "==> job $job $state (exit ${code:-?}); log follows"
rngd logs "$job" || true

[ "${code:-1}" = "0" ]
