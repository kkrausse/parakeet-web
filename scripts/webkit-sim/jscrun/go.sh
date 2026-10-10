# usage: go.sh BUILD PASSES [jsc options...] -> pass times, then wall / CPU / peak memory of the jsc process
J=/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc
cd "$(dirname "$0")"; b=$1; n=$2; shift; shift
/usr/bin/time -l $J -m "$@" run.mjs -- $b $n 2>&1 | grep -E "load|pass|done|FAILED|wasm-function|rror|real|peak memory footprint|maximum resident" | cut -c1-260
