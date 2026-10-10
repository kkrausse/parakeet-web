# usage: go.sh QUERY WAIT NAME   -> opens the published page with QUERY, waits, screenshots, dumps the page's localStorage step trail
U=C86CCB3B-ABF0-47C0-9722-181B89E510C6
B=${BASE:-https://raspberrypi.guineafowl-truck.ts.net/artifacts/parakeet-ggml-browser/}
xcrun simctl openurl $U "$B?$1"; sleep "$2"
xcrun simctl io $U screenshot "$3.png" >/dev/null 2>&1
sh ls.sh
