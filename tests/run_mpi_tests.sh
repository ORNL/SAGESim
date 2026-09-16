#!/usr/bin/env bash
# Run SAGESim's multi-rank tests across several rank counts.
#
# Plain `pytest` runs single-rank only, so nothing here is covered by the default
# test run -- which is how a release shipped with get_agent_property_value
# answering from ghost rows at every rank count > 1.
#
# One GPU is enough: --oversubscribe lets N ranks share it, and these networks are
# tiny. Each file runs at each rank count separately, because a rank count changes
# the partition and some failures only appear at one of them.
#
# Usage:
#   scripts/run_mpi_tests.sh                # ranks 1 2 3 4
#   scripts/run_mpi_tests.sh 2 4            # only those rank counts
#   PYTHON=/path/to/python scripts/run_mpi_tests.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2

PYTHON="${PYTHON:-python}"
TIMEOUT="${TIMEOUT:-600}"
RANKS=("${@:-1 2 3 4}")
read -r -a RANKS <<< "${RANKS[*]}"

FILES=(
    tests/test_worker_sync.py
    tests/test_ghost_readback.py
)

printf '%-34s' "test"
for n in "${RANKS[@]}"; do printf '%10s' "n=$n"; done
echo

failed=0
for f in "${FILES[@]}"; do
    printf '%-34s' "$(basename "$f")"
    for n in "${RANKS[@]}"; do
        if [ "$n" -eq 1 ]; then
            out=$(timeout "$TIMEOUT" "$PYTHON" -m pytest "$f" -q 2>&1)
        else
            out=$(timeout "$TIMEOUT" mpirun --oversubscribe -n "$n" \
                  "$PYTHON" -m pytest "$f" -q 2>&1)
        fi
        rc=$?
        if [ $rc -eq 124 ]; then
            printf '%10s' "TIMEOUT"; failed=1
        elif [ $rc -ne 0 ]; then
            printf '%10s' "FAIL"; failed=1
            printf '\n%s\n' "$out" | sed -e 's/\x1b\[[0-9;]*m//g' \
                | grep -E '^(FAILED|ERROR)' | sort -u | sed 's/^/      /'
            printf '%-34s' ""
        else
            # e.g. "4 passed" / "2 passed, 1 skipped"
            printf '%10s' "$(printf '%s' "$out" | sed -e 's/\x1b\[[0-9;]*m//g' \
                | grep -oE '[0-9]+ passed' | head -1 | cut -d' ' -f1)ok"
        fi
    done
    echo
done

if [ $failed -ne 0 ]; then
    echo
    echo "FAILURES above. A rank-0-only readback deadlocks instead of failing:"
    echo "Model.get_agent_property_value is collective, so every rank must call it."
    exit 1
fi
echo
echo "All MPI tests passed at ranks: ${RANKS[*]}"
