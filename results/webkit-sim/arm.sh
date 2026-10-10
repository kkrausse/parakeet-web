# usage: arm.sh SITE QUERY WAIT   -> CPU-fallback run of the page served from the Mac in simulator Safari (ARM64 JavaScriptCore)
U=C86CCB3B-ABF0-47C0-9722-181B89E510C6
xcrun simctl openurl $U "http://localhost:8793/$1/?$2&auto=1"; sleep "$3"
LAST=1 sh ls.sh | grep -E "RUN|warm|first run|ERROR|transcript" | grep -v "^UA"
