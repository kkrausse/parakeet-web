U=C86CCB3B-ABF0-47C0-9722-181B89E510C6
xcrun simctl terminate $U com.apple.mobilesafari 2>/dev/null
P=$(find ~/Library/Developer/CoreSimulator/Devices/$U/data/Containers/Data/Application -name com.apple.mobilesafari.plist | head -1)
for k in WebKitExperimentalWebGPUEnabled WebKitFeatureWebGPUEnabled WebKitWebGPUEnabled WebKitPreferences.WebGPUEnabled WebKitPreferences.webGPUEnabled WebKitDebugWebGPUEnabled WebKitInternalWebGPUEnabled; do
  xcrun simctl spawn $U defaults write "${P%.plist}" "$k" -bool true
done
xcrun simctl spawn $U defaults read "${P%.plist}" | grep -i gpu
