"""Check preliminary association and bridge serial ordering on a virtual device."""
import importlib.util
from pathlib import Path
import sys
import time


def run_association(server):
    sys.path.insert(0, str(server))
    from src.gimbal import live_tracks
    match = live_tracks._preliminary_box
    prediction = (280., 180., 360., 330.)
    person = [288., 183., 368., 333., .9, 0]
    assert match(prediction, [person]) == tuple(person[:4])
    assert match(prediction, [person, [325., 180., 405., 330., .8, 0]]) is None
    assert match(prediction, [[250., 140., 403., 374., .95, 0]]) is None
    assert match(prediction, [[288., 183., 368., 333., .2, 0]]) is None
    print("association: stable match, crowd, expanded box and weak detection passed")


def run_bridge(server, executable):
    spec = importlib.util.spec_from_file_location("rig", server / "src/gimbal/test_gimbal_bridge.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.BRIDGE = str(executable)
    r = fixture.Rig("--arrival-control", "1", "--heartbeat-ms", "1500",
                    "--observation-grace-ms", "20", "--deadband", "0.05")
    try:
        r.wait_line("PRELIMINARY_CONTROL 1")
        r.send("P")
        r.frame_id += 1
        r.send(f"F {r.frame_id} {time.monotonic_ns()} 1 7 310 200 40 80 .9 OBS 1")
        time.sleep(.012)
        r.send("W 7")
        assert "OK" in r.wait_line("R W ")
        epoch = int(r.wait_line("CONTROL_EPOCH ").split()[1])
        r.wait_state({"auto": 1, "id": 7})
        while r.serial(.003) is not None:
            pass
        stamp = time.monotonic_ns()
        r.send(f"E {stamp} 7 {epoch + 1} 355 240")
        assert r.serial(.010) is None, "stale selection epoch moved the gimbal"
        r.send(f"E {time.monotonic_ns()} 7 {epoch} 420 240")
        assert r.serial(.010) is None, "large correction moved the gimbal"
        stamp = time.monotonic_ns()
        r.send(f"E {stamp} 7 {epoch} 355 240")
        before = r.serial(.020)
        assert before and before["T"] == 141 and before["X"] == 1, before
        first_ms = (time.monotonic_ns() - stamp) / 1e6
        r.frame_id += 1
        r.send(f"F {r.frame_id} {stamp} 1 7 300 200 40 80 .9 OBS 1")
        assert r.until_stop(.15), "verified frame did not correct provisional motion"
        print(f"bridge: preliminary right command {first_ms:.2f} ms after offer, then verified stop")
    finally:
        r.close()


if __name__ == "__main__":
    server = Path(__file__).resolve().parents[2]
    executable = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1
                  else server / "src/gimbal/gimbal_bridge_v27")
    run_association(server)
    run_bridge(server, executable)
