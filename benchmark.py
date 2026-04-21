#!/usr/bin/env python3
"""
Benchmark installed guide-llm-* Ollama models on a given prompt.

Measures tokens/sec, time-to-first-token, total latency, tokens generated,
and peak GPU utilization / memory usage per model.

GPU monitoring:
  - On Jetson (detected via /etc/nv_tegra_release): parses `tegrastats`.
  - Elsewhere (x86 + discrete NVIDIA): uses `nvidia-smi`.

Usage:
    python3 benchmark.py "take me to the kitchen"
    python3 benchmark.py "describe my surroundings" --runs 3
    python3 benchmark.py "hello" --models guide-llm-2b-q8 guide-llm-4b-q4
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time
import urllib.request


IS_JETSON = os.path.exists("/etc/nv_tegra_release")


def poll_tegrastats(stop_event, out):
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "200"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except FileNotFoundError:
        return
    gpu_re = re.compile(r"GR3D_FREQ\s+(\d+)%")
    ram_re = re.compile(r"RAM\s+(\d+)/\d+MB")
    try:
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            m_gpu = gpu_re.search(line)
            m_ram = ram_re.search(line)
            if m_gpu:
                out["peak_util"] = max(out["peak_util"], int(m_gpu.group(1)))
            if m_ram:
                out["peak_mem"] = max(out["peak_mem"], int(m_ram.group(1)))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def poll_nvidia_smi(stop_event, out):
    while not stop_event.is_set():
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1,
            )
            if res.returncode == 0:
                parts = res.stdout.strip().split(",")
                util, mem = int(parts[0].strip()), int(parts[1].strip())
                out["peak_util"] = max(out["peak_util"], util)
                out["peak_mem"] = max(out["peak_mem"], mem)
        except Exception:
            pass
        time.sleep(0.2)


def poll_gpu(stop_event, out):
    if IS_JETSON:
        poll_tegrastats(stop_event, out)
    else:
        poll_nvidia_smi(stop_event, out)


UNLOAD_SETTLE_SEC = 3


def unload(model):
    """Ask Ollama to unload the model from GPU (keep_alive=0)."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body, headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        pass


def run_one(model, prompt):
    gpu = {"peak_util": 0, "peak_mem": 0}
    stop = threading.Event()
    t = threading.Thread(target=poll_gpu, args=(stop, gpu))
    t.start()

    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body, headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())

    stop.set()
    t.join()

    return {
        "tokens": r["eval_count"],
        "tok_s": r["eval_count"] / (r["eval_duration"] / 1e9),
        "ttft_ms": r["prompt_eval_duration"] / 1e6,
        "total_s": r["total_duration"] / 1e9,
        "gpu_pct": gpu["peak_util"],
        "mem_mib": gpu["peak_mem"],
    }


def discover_models():
    res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    return [line.split()[0] for line in res.stdout.splitlines()[1:]
            if line.startswith("guide-llm-")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", help="Prompt to send to each model")
    p.add_argument("--models", nargs="+", help="Model names (default: all installed guide-llm-*)")
    p.add_argument("--runs", type=int, default=1, help="Runs per model, averaged (default 1)")
    args = p.parse_args()

    models = args.models or discover_models()
    if not models:
        raise SystemExit("No guide-llm-* models found. Run: bash setup_ollama.sh all")

    preview = args.prompt if len(args.prompt) <= 60 else args.prompt[:57] + "..."
    print(f"\nPrompt: {preview}")
    print(f"Runs per model: {args.runs}")
    print(f"GPU backend: {'tegrastats (Jetson)' if IS_JETSON else 'nvidia-smi'}\n")
    header = f"{'MODEL':<24}{'TOKENS':>8}{'TOK/S':>8}{'TTFT(ms)':>10}{'TOTAL(s)':>10}{'GPU%':>8}{'MEM(MiB)':>12}"
    print(header)
    print("-" * len(header))

    for m in models:
        # Unload everything first so MEM reflects only this model
        for other in models:
            unload(other)
        time.sleep(UNLOAD_SETTLE_SEC)

        try:
            run_one(m, args.prompt)  # warm-up (discarded) to exclude cold-load from timings
            runs = [run_one(m, args.prompt) for _ in range(args.runs)]
        except Exception as e:
            print(f"{m:<24}  ERROR: {e}")
            continue
        avg = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
        print(f"{m:<24}{avg['tokens']:>8.0f}{avg['tok_s']:>8.1f}{avg['ttft_ms']:>10.0f}"
              f"{avg['total_s']:>10.2f}{avg['gpu_pct']:>8.0f}{avg['mem_mib']:>12.0f}")

    # Final cleanup so no model is left warm
    for m in models:
        unload(m)


if __name__ == "__main__":
    main()
