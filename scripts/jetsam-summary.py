import json, sys
for p in sys.argv[1:]:
    s = open(p).read()
    head, body = s.split("\n", 1)
    h = json.loads(head)
    print("=====", p.split("/")[-1], h.get("timestamp"), h.get("bug_type"), h.get("os_version"))
    try:
        b = json.loads(body)
    except Exception as e:
        print(body[:3000]); continue
    if "processes" not in b:
        print(json.dumps(b)[:3000]); continue
    ps = b.get("pageSize", 16384)
    print(" largestProcess:", b.get("largestProcess"), "pageSize", ps, "memoryStatus keys:", {k: v for k, v in b.get("memoryStatus", {}).items() if not isinstance(v, dict)})
    print(" memoryPages:", b.get("memoryStatus", {}).get("memoryPages"))
    procs = b["processes"]
    for x in procs:
        if "reason" in x:
            print("  KILLED:", x.get("name"), "pid", x.get("pid"), "reason", x.get("reason"), "rpages", x.get("rpages"), "= %.0f MiB" % (x.get("rpages", 0) * ps / 2**20), "lifetimeMax %.0f MiB" % (x.get("lifetimeMax", 0) * ps / 2**20), "states", x.get("states"), "killDelta", x.get("killDelta"), "memlimit?", x.get("mem_regions"), x.get("coalition"))
    top = sorted(procs, key=lambda x: -x.get("rpages", 0))[:8]
    for x in top:
        print("  top:", x.get("name"), "pid", x.get("pid"), "%.0f MiB" % (x.get("rpages", 0) * ps / 2**20), "lifetimeMax %.0f MiB" % (x.get("lifetimeMax", 0) * ps / 2**20), x.get("states"), x.get("reason", ""))
    for x in procs:
        n = x.get("name", "")
        if "WebKit" in n or "Brave" in n or "Safari" in n:
            print("  web:", n, "pid", x.get("pid"), "%.0f MiB" % (x.get("rpages", 0) * ps / 2**20), "lifetimeMax %.0f MiB" % (x.get("lifetimeMax", 0) * ps / 2**20), x.get("states"), x.get("reason", ""), "age_s", (x.get("age", 0) / 1e9 if x.get("age") else None))
