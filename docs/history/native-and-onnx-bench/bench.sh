#!/bin/sh
# Runs run-all.sh as a transient systemd user unit (nice 0, like the dictation service) in its own
# slice with a high CPU weight, so other work on the box disturbs the timings less. The weight only
# wins against sibling cgroups of the user manager; login-session scopes still get their share.
# usage: ./bench.sh [name ...]
cd "$(dirname "$0")"
systemctl --user start parakeetbench.slice
systemctl --user set-property --runtime parakeetbench.slice CPUWeight=10000
exec systemd-run --user --wait --pipe --collect --quiet --slice=parakeetbench.slice -p CPUWeight=10000 -p Nice=0 \
  -E PATH="$PATH" -E HOME="$HOME" -E RUNS="${RUNS:-10}" --working-directory="$PWD" ./run-all.sh "$@"
