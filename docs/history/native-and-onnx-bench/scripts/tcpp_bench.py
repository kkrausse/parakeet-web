"""transcribe.cpp 0.3.1 (ggml, Vulkan) baseline, in its own process.

    python3 scripts/tcpp_bench.py GGUF [--stream] [--runs 10] [--backend vulkan|cpu] AUDIO.f32 ...

Reuses the ctypes bindings of ../dictation-server-linux/server.py and the libraries the live service
uses (read-only). Default mode is offline: one transcribe_run() per clip, with the library's own
mel/encode/decode split. --stream runs the service's unit of work instead: buffered streaming at
(left 5.6 s, chunk 560 ms, right 560 ms), fed back to back in 560 ms pieces, timing every feed.
Prints one JSON object per line, in the same shape as ort-bench.
"""

import ctypes
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

START = time.monotonic()
HERE = Path(__file__).resolve().parent
LIBRARY = Path(os.environ.get("TCPP_LIB", Path.home() / "devfs/repos/kkrausse/random/dictation-server-linux/.cache/transcribe-native-0.3.1"))

spec = importlib.util.spec_from_file_location("server", HERE.parent.parent / "dictation-server-linux/server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class Timings(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint64), ("load_ms", ctypes.c_float), ("mel_ms", ctypes.c_float),
                ("encode_ms", ctypes.c_float), ("decode_ms", ctypes.c_float)]


def ms(since):
    return (time.monotonic() - since) * 1e3


def gpu_mib():
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.split("\n")
    return sum(int(r.split(",")[1]) for r in rows if r.strip() and int(r.split(",")[0]) == os.getpid())


def rss_mib(key="VmRSS:"):
    for line in Path("/proc/self/status").read_text().split("\n"):
        if line.startswith(key):
            return round(int(line.split()[1]) / 1024)


def stats(values):
    if not values:
        return None
    return {"median": round(statistics.median_high(values), 1), "worst": round(max(values), 1), "best": round(min(values), 1)}


def main():
    args = sys.argv[1:]
    stream = "--stream" in args
    args = [a for a in args if a != "--stream"]
    named = {}
    while "--runs" in args or "--backend" in args:
        i = next(i for i, a in enumerate(args) if a in ("--runs", "--backend"))
        named[args[i]] = args[i + 1]
        del args[i:i + 2]
    runs = int(named.get("--runs", 10))
    backend = named.get("--backend", "vulkan")
    weights, clips = args[0], args[1:]

    lib = server.load_library(LIBRARY)
    assert lib.transcribe_abi_struct_size(5) == ctypes.sizeof(Timings)
    lib.transcribe_run.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
    lib.transcribe_get_timings.argtypes = [ctypes.c_void_p, ctypes.POINTER(Timings)]
    engine = server.SimpleNamespace(lib=lib, session=ctypes.c_void_p(), chunk_ms=560, right_ms=560, backend="")

    t = time.monotonic()
    init = server.BackendInitParams()
    lib.transcribe_backend_init_params_init(ctypes.byref(init))
    init.artifact_dir = str(LIBRARY).encode()
    init.allowed_backends = server.BACKEND_MASK_CPU | (server.BACKEND_MASK_VULKAN if backend == "vulkan" else 0)
    server.check(engine, lib.transcribe_init_backends_ex(ctypes.byref(init)), "init_backends")
    init_ms = ms(t)
    t = time.monotonic()
    load = server.ModelLoadParams()
    lib.transcribe_model_load_params_init(ctypes.byref(load))
    load.backend = server.BACKEND_VULKAN if backend == "vulkan" else 1  # TRANSCRIBE_BACKEND_CPU
    model = ctypes.c_void_p()
    server.check(engine, lib.transcribe_model_load_file(weights.encode(), ctypes.byref(load), ctypes.byref(model)), "model_load_file")
    params = server.SessionParams()
    lib.transcribe_session_params_init(ctypes.byref(params))
    server.check(engine, lib.transcribe_session_init(model, ctypes.byref(params), ctypes.byref(engine.session)), "session_init")
    load_ms = ms(t)
    time.sleep(0.25)
    print(json.dumps({"record": "load", "runtime": f"transcribe.cpp {lib.transcribe_version().decode()}",
                      "backend": lib.transcribe_model_backend(model).decode(), "weights": Path(weights).name,
                      "mode": "stream" if stream else "offline",
                      "load_ms": {"backends_init": round(init_ms), "model_and_session": round(load_ms),
                                  "process_start_to_loaded": round(ms(START))},
                      "gpu_mib_after_load": gpu_mib(), "rss_mib_after_load": rss_mib()}), flush=True)

    def offline(audio):
        t = time.monotonic()
        server.check(engine, lib.transcribe_run(engine.session, audio, len(audio) // 4, None), "run")
        text = lib.transcribe_full_text(engine.session).decode()
        total = ms(t)
        timings = Timings()
        lib.transcribe_timings_init(ctypes.byref(timings))
        lib.transcribe_get_timings(engine.session, ctypes.byref(timings))
        return {"pre": timings.mel_ms, "enc": timings.encode_ms, "dec": timings.decode_ms, "total": total, "text": text.strip()}

    def streamed(audio):
        """Returns per-feed times for feeds that ran inference with a full left context."""
        t = time.monotonic()
        server.reset(engine)
        chunks = []
        piece = 560 * 64
        for offset in range(0, len(audio), piece):
            c = time.monotonic()
            server.feed(engine, audio[offset:offset + piece])
            chunks.append((offset / 64000, ms(c)))
        text = server.finish(engine)
        # A chunk at audio time T is encoded with min(T, 5.6 s) of left context; keep the full windows.
        full = [m for at, m in chunks if at >= 5.6 + 0.56]
        return {"chunks": [round(m, 1) for _, m in chunks], "full": full, "total": ms(t), "text": text.strip()}

    peak = 0
    for i, path in enumerate(clips):
        audio = Path(path).read_bytes()
        seconds = len(audio) / 64000
        run = streamed if stream else offline
        first = run(audio)
        since_start = ms(START)
        results = []
        if runs:
            run(audio)
            results = [run(audio) for _ in range(runs)]
        peak = max(peak, gpu_mib())
        record = {"record": "audio", "file": path, "seconds": round(seconds, 2), "first_in_process": i == 0,
                  "process_start_to_first_transcript_ms": round(since_start) if i == 0 else None,
                  "warm_runs": len(results), "total_ms": stats([r["total"] for r in results]),
                  "gpu_mib_peak": peak, "rss_mib_peak": rss_mib("VmHWM:"),
                  "text": first["text"], "text_stable": all(r["text"] == first["text"] for r in results)}
        if results:
            record["rtf_x_realtime"] = round(seconds * 1e3 / record["total_ms"]["median"], 1)
        if stream:
            record["first_run_ms"] = {"chunks": first["chunks"], "total": round(first["total"])}
            record["full_window_chunk_ms"] = stats([m for r in results for m in r["full"]])
            record["full_window_chunks_per_run"] = len(first["full"])
        else:
            record["first_run_ms"] = {k: round(first[k]) for k in ("pre", "enc", "dec", "total")}
            for key in ("pre", "enc", "dec"):
                record[f"{key}_ms"] = stats([r[key] for r in results])
        print(json.dumps(record), flush=True)


main()
os._exit(0)
