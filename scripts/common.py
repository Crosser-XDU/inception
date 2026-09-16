"""Shared validation for the exported, explicit-route launchers."""

import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "LLaMA-Factory"
EXPERIMENT = REPO / "experiments/recurft_math"


def environment(gpu=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(map(str, [REPO / "src", REPO, EXPERIMENT]))
    env["DISABLE_VERSION_CHECK"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def checkpoint_metadata(path, route):
    path = Path(path)
    required = ["adapter_config.json", "adapter_model.safetensors", "recurft_config.json",
                "recurft_recurrent.safetensors", "tokenizer_config.json", "tokenizer.json"]
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f"Missing checkpoint file: {path / name}")
    metadata = json.loads((path / "recurft_config.json").read_text())
    if route == "boundary" and metadata.get("boundary_head_rank", 0) <= 0:
        raise ValueError("This checkpoint has no trained boundary head; select tail or supply a boundary checkpoint.")
    if not metadata.get("projection_layer_ids"):
        raise ValueError("Checkpoint lacks explicit projection_layer_ids.")
    return metadata


def validate_data(path, start, samples):
    if start < 0 or samples <= 0:
        raise ValueError("Require start >= 0 and samples > 0.")
    path = Path(path)
    if path.suffix != ".jsonl":
        raise ValueError("The portable launcher requires JSONL with question and answer fields.")
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line {line_number}; physical row indices must be stable.")
            row = json.loads(line)
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"Missing/non-string question at line {line_number}")
            if "answer" not in row:
                raise ValueError(f"Missing answer at line {line_number}")
            rows.append(row)
    selected = rows[start:start + samples]
    if len(selected) != samples:
        raise ValueError(f"Requested {samples} samples starting at {start}, but data has {len(rows)} rows.")
    if len({r["question"] for r in selected}) != samples:
        raise ValueError("Selected questions are not unique.")
    return {"total_rows": len(rows), "selected_rows": samples, "sha256": sha256(path)}


def check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numeric value in result JSON.")
    if isinstance(value, dict):
        for child in value.values():
            check_finite(child)
    elif isinstance(value, list):
        for child in value:
            check_finite(child)


def audit_correction_mode(summary, rows, mode):
    """Check requested switches and execution counters without claiming a speedup."""
    expected = {"off": (False, False), "reuse": (True, False), "defer": (False, True)}
    if mode not in expected:
        raise ValueError(f"Unknown correction mode: {mode}")
    switches = ("reuse_verify_cache_for_correction", "defer_correction_to_next_verify")
    for name, enabled in zip(switches, expected[mode]):
        if summary["args"].get(name, False) is not enabled:
            raise ValueError(f"Correction mode {mode} does not match result flag {name}.")
    counters = {}
    for name in ("fast_correction_cache_reuses", "deferred_corrective_tokens"):
        value = summary.get(name)
        per_sample = [row["adaptive"].get(name) for row in rows]
        if any(type(item) is not int or item < 0 for item in [value, *per_sample]):
            raise ValueError(f"Missing or invalid correction counter: {name}")
        if sum(per_sample) != value:
            raise ValueError(f"Correction counter does not match per-sample totals: {name}")
        counters[name] = value
    reuse = counters["fast_correction_cache_reuses"]
    deferred = counters["deferred_corrective_tokens"]
    if (mode != "reuse" and reuse) or (mode != "defer" and deferred):
        raise ValueError("An unrequested correction branch executed.")
    return {"correction_mode": mode, "correction_optimization_observed": bool(reuse or deferred), **counters}


def audit_draft_policy(summary, rows, expected):
    """Confirm the policy and aggregate blocks; gate and cooldown skips may overlap."""
    for key, value in expected.items():
        if summary["args"].get(key) != value:
            raise ValueError(f"Draft policy does not match result flag {key}.")
    histogram = {}
    for row in rows:
        hist = row["adaptive"].get("block_hist")
        if not isinstance(hist, dict):
            raise ValueError("Missing block_hist; update the decoder to report adaptive execution.")
        for length, count in hist.items():
            if not str(length).isdigit() or int(length) < 1 or type(count) is not int or count < 0:
                raise ValueError("Invalid draft block histogram.")
            histogram[str(int(length))] = histogram.get(str(int(length)), 0) + count
    blocks = sum(histogram.values())
    target_only = histogram.get("1", 0)
    draft_blocks = blocks - target_only
    drafts = sum((int(length) - 1) * count for length, count in histogram.items())
    if drafts != summary["draft_tokens"]:
        raise ValueError("Draft block histogram does not match draft_tokens.")
    accepted = summary["accepted_draft_tokens"]
    if type(accepted) is not int or not 0 <= accepted <= drafts:
        raise ValueError("Invalid accepted draft count.")
    counters = {}
    for key in ("cheap_policy_evaluations", "cheap_policy_skips",
                "draft_cooldown_skips", "draft_cooldown_activations"):
        counts = [summary.get(key), *(row["adaptive"].get(key) for row in rows)]
        if any(type(n) is not int or n < 0 for n in counts) or counts[0] != sum(counts[1:]):
            raise ValueError(f"Missing or inconsistent adaptive counter: {key}")
        counters[key] = counts[0]
    if counters["cheap_policy_skips"] > counters["cheap_policy_evaluations"]:
        raise ValueError("Gate skips exceed gate evaluations.")
    phase_stats = {}
    for phase in ("early", "mid", "late"):
        phase_blocks = phase_drafted_blocks = phase_drafts = phase_accepted = reaching_samples = 0
        for row in rows:
            a = row["adaptive"]
            hist = a.get("block_hist_by_phase", {}).get(phase, {})
            accepted_hist = a.get("accepted_hist_by_phase", {}).get(phase, {})
            n = sum(hist.values())
            phase_blocks += n
            reaching_samples += int(n > 0)
            phase_drafted_blocks += sum(count for length, count in hist.items() if int(length) > 1)
            phase_drafts += sum((int(length) - 1) * count for length, count in hist.items())
            phase_accepted += sum(max(0, int(length) - 1) * count for length, count in accepted_hist.items())
        if phase_blocks:
            phase_stats[phase] = {
                "reaching_samples": reaching_samples, "blocks": phase_blocks,
                "draft_tokens": phase_drafts, "accepted_draft_tokens": phase_accepted,
                "draft_acceptance": phase_accepted / phase_drafts if phase_drafts else None,
                "mean_accepted_drafts_per_draft_block": (
                    phase_accepted / phase_drafted_blocks if phase_drafted_blocks else None),
            }
    return {
        "phase_draft_stats": phase_stats,
        "draft_policy": "adaptive" if expected["mode"] == "heuristic" else expected["mode"],
        "runtime_mode": "throughput" if expected.get("production_async_timing") else "diagnostic",
        "draft_policy_settings": expected,
        "recurrent_drafting_observed": drafts > 0,
        "verified_blocks": blocks,
        "draft_blocks": draft_blocks,
        "target_only_blocks": target_only,
        "target_only_block_rate": target_only / blocks if blocks else None,
        "mean_drafts_per_draft_block": drafts / draft_blocks if draft_blocks else None,
        "mean_accepted_drafts_per_draft_block": accepted / draft_blocks if draft_blocks else None,
        "mean_accepted_drafts_per_block": accepted / blocks if blocks else None,
        "draft_block_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        **counters,
    }


def audit_result(path, route, start, samples, correction_mode=None, draft_policy=None):
    data = json.loads(Path(path).read_text())
    check_finite(data)
    s, rows = data["summary"], data["results"]
    if s["samples"] != samples or [r["sample"] for r in rows] != list(range(start, start + samples)):
        raise ValueError("Result coverage/order does not match the requested sample interval.")
    args, times = s["args"], s["adaptive_timing_sums"]
    if args.get("draft_commit_policy") != "target_match" or args.get("target_match_lambda") != 1.0:
        raise ValueError("The run did not use strict target-match verification.")
    if any(args.get("target_match_lambda_" + phase) not in (None, 1.0) for phase in ("early", "mid", "late")):
        raise ValueError("A position-specific threshold relaxed target-match verification.")
    for key in ("accepted_mismatch_tokens", "accepted_unchecked_tokens"):
        if key not in s or any(key not in r["adaptive"] for r in rows):
            raise ValueError(f"Missing safety counter: {key}")
        if s[key] != 0 or any(r["adaptive"][key] != 0 for r in rows):
            raise ValueError(f"Unsafe committed token count: {key}")
    policy = audit_draft_policy(s, rows, draft_policy) if draft_policy is not None else {}
    drafts, ngrams = s["draft_tokens"], s.get("ngram_draft_tokens", 0)
    if route == "ngram":
        if args.get("ngram_draft_mode") != "only" or drafts != ngrams:
            raise ValueError("Expected n-gram-only execution.")
        if any(times.get(k, 0) != 0 for k in ("t_step_s", "tail_s", "boundary_s")):
            raise ValueError("Unexpected neural drafting in n-gram control.")
    else:
        if args.get("ngram_draft_mode") != "off" or args["draft_logit_source"] != route:
            raise ValueError("Requested neural draft route was overridden.")
        other = "tail_s" if route == "boundary" else "boundary_s"
        all_skipped = drafts == 0 and (
            policy.get("draft_policy") in {"adaptive", "schedule"}
            or policy.get("draft_policy_settings", {}).get("latent_draft_min_position", 0) > 0
        )
        if all_skipped:
            if ngrams != 0 or any(times.get(k, 0) != 0 for k in ("t_step_s", "tail_s", "boundary_s")):
                raise ValueError("Zero draft count conflicts with recorded drafting work.")
        else:
            neural_time = sum(times.get(key, 0) for key in ("t_step_s", "t_init_s", "t_sync_s"))
            parallel_time = times.get("draft_parallel_s", 0) if args.get("single_gpu_parallel_draft") else 0
            if drafts <= 0 or ngrams != 0 or neural_time + parallel_time <= 0:
                raise ValueError("No evidence that recurrent drafting actually ran.")
            if times.get(route + "_s", 0) + parallel_time <= 0 or times.get(other, 0) != 0:
                raise ValueError("Projection timing does not match the selected route.")
    correction = audit_correction_mode(s, rows, correction_mode) if correction_mode is not None else {}
    return {**correction, **policy, "route": route, "samples": samples, "result_sha256": sha256(path),
            "wall_speedup": s["wall_clock_speedup"], "target_call_speedup": s["target_call_speedup"],
            "baseline_correct": sum(r["baseline_correct"] is True for r in rows),
            "recurrent_correct": sum(r["correct"] is True for r in rows),
            "wrong_to_right": sum(r["baseline_correct"] is False and r["correct"] is True for r in rows),
            "right_to_wrong": sum(r["baseline_correct"] is True and r["correct"] is False for r in rows),
            "draft_tokens": drafts, "ngram_draft_tokens": ngrams,
            "component_times": times, "numerics": "finite", "quality_gate_applied": False}


def gpu_snapshot(gpu):
    def query(fields, compute=False):
        kind = "compute-apps" if compute else "gpu"
        cmd = ["nvidia-smi", "-i", str(gpu), f"--query-{kind}={fields}", "--format=csv,noheader,nounits"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
        return [r for r in csv.reader(io.StringIO(result.stdout), skipinitialspace=True) if r]
    rows = query("uuid,memory.free,utilization.gpu")
    if len(rows) != 1:
        raise ValueError("Select exactly one physical GPU index or UUID.")
    uuid, free, util = rows[0]
    pids = [int(r[0]) for r in query("pid", compute=True)]
    return {"time": time.time(), "uuid": uuid, "free_mib": int(free), "util": int(util), "compute_pids": pids}


class GPUReservation:
    """Cooperative launcher lock plus sampled observation, never kills other jobs."""
    def __init__(self, gpu):
        self.gpu = gpu
        self.handle = None
        self.initial = None

    def __enter__(self):
        if self.gpu is None:
            return self
        state = gpu_snapshot(self.gpu)
        path = Path(tempfile.gettempdir()) / f"layerloop-{os.getuid()}-{state['uuid']}.lock"
        self.handle = path.open("a")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = gpu_snapshot(self.gpu)
            if state["free_mib"] < 51200 or state["util"] > 20 or state["compute_pids"]:
                raise RuntimeError(f"GPU not eligible: {state}. No process was changed.")
        except Exception:
            self.handle.close()
            self.handle = None
            raise
        self.initial = state
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.close()


def run_logged(cmd, output, gpu=None):
    output = Path(output)
    with GPUReservation(gpu) as reservation:
        observations = [reservation.initial] if gpu is not None else []
        errors = []
        stop = threading.Event()
        with (output / "run.log").open("x") as log:
            child = subprocess.Popen(cmd, env=environment(gpu), cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            write_json(output / "started.marker.json", {"launcher_pid": os.getpid(), "compute_pid": child.pid,
                       "command": cmd, "gpu_preflight": reservation.initial})
            def monitor():
                while not stop.wait(5):
                    try:
                        observations.append(gpu_snapshot(gpu))
                    except Exception as e:
                        errors.append(str(e))
            thread = threading.Thread(target=monitor, daemon=True) if gpu is not None else None
            if thread:
                thread.start()
            try:
                code = child.wait()
            except BaseException:
                # This is our exact child, not a name-based process search.
                child.terminate()
                child.wait()
                raise
            finally:
                stop.set()
                if thread:
                    thread.join()
                foreign = sorted({p for s in observations for p in s["compute_pids"] if p != child.pid})
                timing = "cpu_correctness_only" if gpu is None else (
                    "provisional_overlap_or_monitor_error" if foreign or errors else "sampled_exclusive")
                report = {"timing_status": timing, "foreign_compute_pids": foreign,
                          "monitor_errors": errors, "observations": observations,
                          "caveat": "Five-second samples cannot prove continuous device exclusivity."}
                write_json(output / "gpu_audit.json", report)
        if code:
            raise RuntimeError(f"Child exited {code}; inspect {output / 'run.log'}")
    return report
