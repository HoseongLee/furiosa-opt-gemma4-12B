#!/bin/sh
set -u

here=$(dirname "$0")
cd "$here" || exit 1

if [ ! -f ./fixtures.safetensors ]; then
    echo "run_tests.sh: fixtures.safetensors is missing from the submission" >&2
    exit 1
fi

binary=./test_runtime
chmod +x "$binary" 2>/dev/null || true
if [ ! -x "$binary" ]; then
    echo "run_tests.sh: test_runtime is not executable; running an owned copy"
    cp ./test_runtime ./test_runtime.exec || exit 126
    chmod +x ./test_runtime.exec || exit 126
    binary=./test_runtime.exec
fi

"$binary"
status=$?
rm -f ./test_runtime.exec 2>/dev/null || true

if [ "$status" -eq 126 ] || [ "$status" -eq 127 ]; then
    echo "run_tests.sh: could not execute test_runtime at all (exit $status)" >&2
fi

if [ "$status" -eq 0 ]; then
    echo "run_tests.sh: all kernel tests passed"
else
    echo "run_tests.sh: kernel tests failed (exit $status)" >&2
fi
exit "$status"
