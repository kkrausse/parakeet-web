// Summarise where ONNX Runtime put the encoder's nodes.
// Run: bun scripts/placement.ts VERBOSE_STDERR_LOG [PROFILE_JSON]
// The log comes from `ort-bench --verbose ... 2> log`, the profile from `--profile PREFIX`.
import { readFileSync } from "node:fs";

const [logPath, profilePath] = process.argv.slice(2);
const sessions: { provider: string; count: number; ops: Record<string, number> }[][] = [];
for (const line of readFileSync(logPath!, "utf8").split("\n")) {
  if (!line.includes("VerifyEachNodeIsAssignedToAnEp")) continue;
  if (/\] Node placements/.test(line)) { sessions.push([]); continue; }
  const group = line.match(/placed on \[(\w+)\]\. Number of nodes: (\d+)/);
  if (group) { sessions.at(-1)!.push({ provider: group[1]!, count: Number(group[2]), ops: {} }); continue; }
  const node = line.match(/\]\s+(\w+) \(/);
  const current = sessions.at(-1)?.at(-1);
  if (node && current) current.ops[node[1]!] = (current.ops[node[1]!] ?? 0) + 1;
}
// Sessions are created in this order by ort-bench.
["preprocessor (nemo128)", "encoder", "decoder+joint"].forEach((name, i) => {
  console.log(`\n${name}`);
  for (const group of sessions[i] ?? []) {
    const ops = Object.entries(group.ops).sort((a, b) => b[1] - a[1]).map(([op, n]) => `${op} ${n}`).join(", ");
    console.log(`  ${group.provider}: ${group.count} nodes${ops ? `  (${ops})` : ""}`);
  }
});

if (profilePath) {
  // Kernel time per provider over all encoder runs in the profile. WebGPU kernel time is dispatch
  // time on the CPU side, not GPU time, so read it as "where the host thread spent its time".
  const events = JSON.parse(readFileSync(profilePath, "utf8")) as any[];
  const byProvider: Record<string, { ms: number; calls: number; ops: Record<string, number> }> = {};
  let runs = 0;
  for (const event of events) {
    if (event.name === "model_run") runs++;
    if (event.cat !== "Node" || !event.name.endsWith("_kernel_time")) continue;
    const entry = (byProvider[event.args.provider] ??= { ms: 0, calls: 0, ops: {} });
    entry.ms += event.dur / 1000;
    entry.calls++;
    entry.ops[event.args.op_name] = (entry.ops[event.args.op_name] ?? 0) + event.dur / 1000;
  }
  console.log(`\nencoder profile, ${runs} runs`);
  for (const [provider, entry] of Object.entries(byProvider)) {
    const top = Object.entries(entry.ops).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([op, ms]) => `${op} ${(ms / runs).toFixed(1)}`).join(", ");
    console.log(`  ${provider}: ${(entry.ms / runs).toFixed(1)} ms/run in ${entry.calls / runs} kernel calls/run  (top: ${top})`);
  }
}
