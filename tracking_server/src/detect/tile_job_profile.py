"""One requested split-camera capture, measured in the existing feed process.

Split-feed samples include copy -> queue publication -> consumer read/wait/
take -> collection boundaries. Wall and per-thread CPU clocks have different
origins: compare CPU deltas only within the same named producer or consumer.
Condition.wait returns after the consumer has reacquired its lock and the GIL;
its wall interval cannot distinguish OS scheduling from GIL/lock contention.
Read/wait fields can be None when the request armed after the read began, or
the consumer took an already queued result without waiting.
"""

import json
from pathlib import Path
import threading
import time


SAMPLE_FRAMES = 600
REQUEST_POLL_FRAMES = 30
NPU_FIELDS = ("input_format_us", "h2d_us", "inference_core_all_us",
              "inference_core_0_us", "inference_core_1_us", "inference_core_2_us",
              "d2h_us", "output_format_us", "total_us")


def task_timings(report):
    if not report.valid:
        return None
    return [
        {"task": task.task_name, "cpu_us": task.cpu_task_us,
         "devices": {device: {field: getattr(timing, field) for field in NPU_FIELDS}
                     for device, timing in task.devices.items() if timing.valid}}
        for task in report.tasks if task.valid
    ]


class TileJobProfile:
    def __init__(self, output, runtime):
        self.output = Path(output)
        self.request = self.output / "tile_job_profile.request"
        self.runtime = runtime
        self.enabled_item = None
        self.profiler = None
        self.capture = None
        self.completed = None
        self.until_poll = 1

    def _arm(self):
        try:
            self.request.unlink()
        except FileNotFoundError:
            return
        except OSError as error:
            print(f"TILE PROFILE request: {error!r}", flush=True)
            return
        self.capture = {"clock": "perf_counter_ns", "sample_limit": SAMPLE_FRAMES,
                        "started_unix_ns": time.time_ns(), "frames": []}
        try:
            cfg = self.runtime.Configuration()
            item = cfg.ITEM.PROFILER
            self.profiler = self.runtime.Profiler.get_instance()
            if not cfg.get_enable(item):
                cfg.set_enable(item, True)
                self.enabled_item = (cfg, item)
            self.capture["runtime_profiler_enabled"] = bool(cfg.get_enable(item))
        except Exception as error:
            self.capture["runtime_profiler_error"] = repr(error)
            self.profiler = None

    def begin(self, frame):
        if frame is None or frame < 0:
            return None
        if self.capture is None:
            self.until_poll -= 1
            if self.until_poll > 0:
                return None
            self.until_poll = REQUEST_POLL_FRAMES
            self._arm()
            if self.capture is None:
                return None
        if len(self.capture["frames"]) >= SAMPLE_FRAMES:
            # The last result is waiting for its consumer boundary. Do not
            # enlarge the requested capture while the tracker is occupied.
            return None
        return {"frame": frame, "prep_start_ns": time.perf_counter_ns(), "tiles": []}

    def record_result(self, trace, job_ids, engines):
        unique_jobs = len(set(job_ids)) == len(job_ids)
        if not unique_jobs:
            trace["duplicate_job_ids"] = True
        for tile, job_id, engine in zip(trace["tiles"], job_ids, engines):
            try:
                tile["sdk_latest_npu_us"] = engine.get_npu_inference_time()
                tile["sdk_latest_total_us"] = engine.get_latency()
                if self.profiler is not None and unique_jobs:
                    tile["tasks"] = task_timings(self.profiler.get_job_metrics(job_id))
            except Exception as error:
                tile["metrics_error"] = repr(error)
        trace["metrics_done_ns"] = time.perf_counter_ns()
        self.completed = (job_ids, trace)

    def finish_frame(self, job_ids, arrival, ready_ns, copied_ns, *, handoff=False):
        if self.completed is None or self.completed[0] is not job_ids:
            return
        _, trace = self.completed
        self.completed = None
        trace["arrival_ns"] = round(arrival * 1e9)
        trace["feed_ready_ns"] = ready_ns
        trace["feed_copy_done_ns"] = copied_ns
        self.capture["frames"].append(trace)
        if not handoff and len(self.capture["frames"]) >= SAMPLE_FRAMES:
            self.close()
        return trace if handoff else None

    def finish_handoff(self, trace, *, discarded=False):
        """Finish before writing, so the last frame has the same boundaries."""
        trace["tracker_collect_done_ns"] = time.perf_counter_ns()
        trace["tracker_collect_cpu_ns"] = time.thread_time_ns()
        trace["handoff_discarded"] = discarded
        if (self.capture is not None and len(self.capture["frames"]) >= SAMPLE_FRAMES
                and self.capture["frames"][-1] is trace):
            self.close()

    def close(self):
        if self.capture is None:
            return
        capture, self.capture = self.capture, None
        self.completed, self.profiler = None, None
        if self.enabled_item is not None:
            cfg, item = self.enabled_item
            self.enabled_item = None
            try:
                cfg.set_enable(item, False)
            except Exception as error:
                capture["runtime_profiler_restore_error"] = repr(error)
        filename = self.output / f"tile_job_profile_{capture['started_unix_ns']}.json"
        threading.Thread(target=self._write, args=(filename, capture), daemon=True).start()

    @staticmethod
    def _write(filename, capture):
        tmp = filename.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(capture, separators=(",", ":")), encoding="utf-8")
            tmp.replace(filename)
        except Exception as error:
            print(f"TILE PROFILE write: {error!r}", flush=True)
