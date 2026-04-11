#!/usr/bin/env python3
import json
import subprocess
import datetime
import socket
import os
import sys

def format_time(ns):
    if ns < 1000:
        return f"{ns:.2g} ns"
    elif ns < 1000000:
        return f"{ns/1000:.1f} µs"
    elif ns < 1_000_000_000:
        return f"{ns/1000000:.1f} ms"
    else:
        return f"{ns/1_000_000_000:.2f} s"

def run_benchmarks():
    benchmarks = [
        ("ENGINE", "build/tests/benchmarks/engine_bench"),
        ("EVAL", "build/tests/benchmarks/eval_bench"),
        ("MODEL", "build/tests/benchmarks/model_bench"),
        ("SELF-PLAY", "build/tests/benchmarks/self_play_bench"),
        ("SOLVER", "build/tests/benchmarks/solver_bench"),
        ("TENSOR", "build/tests/benchmarks/tensor_bench"),
        ("TRAINING", "build/tests/benchmarks/training_bench"),
    ]
    
    results = {}
    context = None
    
    for name, path in benchmarks:
        if not os.path.exists(path):
            continue
        try:
            print(f"Running {name}...", file=sys.stderr)
            # Shorten runtime here for benchmarking
            proc = subprocess.run([path, "--benchmark_format=json", "--benchmark_min_time=0.1s"], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            data = json.loads(proc.stdout)
            if not context:
                context = data['context']
            results[name] = data['benchmarks']
        except Exception as e:
            print(f"Failed to run {name}: {e}", file=sys.stderr)
            
    return context, results

def format_rate(rate_per_sec, type_name="items"):
    if type_name == "items":
        # Check if we should guess from the name
        pass # Will handle custom types in output
        
    val = rate_per_sec
    if val >= 1_000_000_000_000:
        return f"{val/1_000_000_000_000:.2f}T/s"
    elif val >= 1_000_000_000:
        return f"{val/1_000_000_000:.2f}B/s"
    elif val >= 1_000_000:
        return f"{val/1_000_000:.2f}M/s"
    elif val >= 1_000:
        return f"{val/1_000:.1f}k/s"
    else:
        return f"{val:.1f}/s"

def main():
    with open("/proc/cpuinfo", "r") as f:
        m_name = "Unknown"
        for line in f:
            if line.startswith("model name"):
                m_name = line.split(":", 1)[1].strip()
                break

    context, results = run_benchmarks()
    
    if not context:
        print("No benchmarks succeeded.", file=sys.stderr)
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M BST")
    host = socket.gethostname()
    cpu_cores = context['num_cpus']
    
    l3_size = 0
    l3_sharing = 0
    for cache in context['caches']:
        if cache['level'] == 3:
            l3_size = cache['size'] // 1024
            l3_sharing = cpu_cores // cache.get('num_sharing', cpu_cores)
            break
            
    out = []
    out.append("================================================================================")
    out.append("  PRO YAMS AI — Benchmark Results")
    out.append(f"  Date: {now}")
    out.append(f"  Host: {host}")
    out.append(f"  CPU:  {m_name} ({cpu_cores} logical cores)")
    if l3_size:
        out.append(f"  L3 Cache: {l3_size} KiB x{l3_sharing}")
    out.append("================================================================================\n")
    
    for section, benches in results.items():
        out.append(f"--- {section} BENCHMARKS " + "-"*(72 - 17 - len(section)))
        out.append("")
        
        last_group = None
        
        for b in benches:
            # e.g. BM_ForwardPass/64
            name = b['name'].replace("BM_", "")
            
            parts = name.split("/")
            group = parts[0]
            sub = parts[1] if len(parts) > 1 else None
            
            ns = b['real_time']
            unit = b.get('time_unit', 'ns')
            if unit == 'us':
                ns *= 1000
            elif unit == 'ms':
                ns *= 1000000
            elif unit == 's':
                ns *= 1000000000
                
            time_str = format_time(ns)
            if ns < 1000:
                pass 
            elif ns < 1_000_000:
                time_str = f"{ns:.0f} ns       ({time_str})"
            
            # Check counters
            throughput = ""
            if 'items_per_second' in b:
                unit = " items/s"
                if "ForwardPass" in group or "BatchInference" in group or "TrainStep" in group:
                    unit = " samples/s"
                elif "Evaluation" in group:
                    unit = " games/s"
                elif "Queue" in group:
                    unit = " items/s"
                throughput = format_rate(b['items_per_second'])[:-2] + unit
            elif 'bytes_per_second' in b:
                kb = b['bytes_per_second'] / 1024
                mb = kb / 1024
                gb = mb / 1024
                if gb > 1:
                    throughput = f"{gb:.2f} GiB/s"
                elif mb > 1:
                    throughput = f"{mb:.2f} MiB/s"
                else:
                    throughput = f"{kb:.2f} KiB/s"
            
            if sub:
                if group != last_group:
                    out.append(f"  {group}")
                
                label = f"    batch={sub}" if "batch=" not in sub else f"    {sub}"
                # mode cases etc
                if group == "TrainingDataExtraction":
                    label = f"    mode={sub}"
                if group == "SoftmaxSample":
                    label = f"  {group}/{sub}"
                    
                if group == "SoftmaxSample" or "Validation" in sub:
                     # SoftmaxSample is printed straight out
                     line = f"  {name:<30} {time_str:>10}       {throughput}"
                     out.append(line.rstrip())
                else:
                     line = f"{label:<30} {time_str:>10}       {throughput}"
                     out.append(line.rstrip())
            else:
                line = f"  {name:<30} {time_str:>10}       {throughput}"
                out.append(line.rstrip())
            
            last_group = group
            
        out.append("")
        
    out.append("================================================================================")
    out.append("  End of benchmark report")
    out.append("================================================================================")
    out.append("")

    with open("benchmark_results_new.log", "w") as f:
        f.write("\n".join(out))
    print("Saved to benchmark_results_new.log")

if __name__ == "__main__":
    main()
