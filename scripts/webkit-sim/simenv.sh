# usage: simenv.sh "JSC_opt=val JSC_opt2=val" QUERY SECONDS -> like simsoak.sh, with JavaScriptCore options for Safari's web content processes
U=C86CCB3B-ABF0-47C0-9722-181B89E510C6
xcrun simctl boot $U 2>/dev/null; xcrun simctl bootstatus $U >/dev/null 2>&1
xcrun simctl terminate $U com.apple.mobilesafari 2>/dev/null; sleep 2
for kv in $1; do export "SIMCTL_CHILD___XPC_$kv"; export "SIMCTL_CHILD_$kv"; done
xcrun simctl launch $U com.apple.mobilesafari >/dev/null; sleep 3
xcrun simctl openurl $U "https://raspberrypi.guineafowl-truck.ts.net/artifacts/parakeet-live/?$2"
t=0
while [ $t -lt "$3" ]; do
  sleep 5; t=$((t+5))
  echo "t=$t $(ps -axo rss=,pid=,comm= | grep -i 'WebKit.WebContent' | sort -rn | head -2 | awk '{printf "%s:%d ", $2, $1/1024}')"
done
ps -axo pid=,rss=,etime=,time=,comm= | grep -i "WebKit.WebContent" | sort -k2 -rn | head -1 | awk '{printf "END pid %s rss=%d MiB etime=%s cpu=%s\n", $1, $2/1024, $3, $4}'
