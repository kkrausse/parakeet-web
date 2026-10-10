# usage: simsoak.sh QUERY SECONDS -> opens the published live page in simulator Safari (restarted first) and samples
# the simulator's WebContent processes (pid:RSS MiB, largest three) every 5 s
U=C86CCB3B-ABF0-47C0-9722-181B89E510C6
xcrun simctl boot $U 2>/dev/null; xcrun simctl bootstatus $U >/dev/null 2>&1
xcrun simctl terminate $U com.apple.mobilesafari 2>/dev/null; sleep 2
xcrun simctl openurl $U "https://raspberrypi.guineafowl-truck.ts.net/artifacts/parakeet-live/?$1"
t=0
while [ $t -lt "$2" ]; do
  sleep 5; t=$((t+5))
  echo "t=$t $(ps -axo rss=,pid=,comm= | grep -i 'WebKit.WebContent' | sort -rn | head -3 | awk '{printf "%s:%d ", $2, $1/1024}')"
done
ps -axo pid=,rss=,etime=,time=,comm= | grep -i "WebKit.WebContent" | sort -k2 -rn | head -2 | awk '{printf "END pid %s rss=%d MiB etime=%s cpu=%s\n", $1, $2/1024, $3, $4}'
