#!/usr/bin/env python3
"""Exercise --catchup-speed over a pseudo-terminal. No physical device is used."""
import subprocess
import time

import test_gimbal_bridge as bridge


def check_option_bounds():
    for invalid in ("-1", "8", "7.0", "7x"):
        result = subprocess.run(
            [bridge.BRIDGE, "--catchup-speed", invalid],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=False,
        )
        assert result.returncode != 0 and b"--catchup-speed must be within 0..7" in result.stderr


def path(option, quality):
    args = [] if option is None else ["--catchup-speed", str(option)]
    rig = bridge.Rig("--controller", "pd", "--kp", "30", "--ki", "0", "--kd", "6",
                     "--aim-y", "0.5", "--deadband", "0.06", "--speed-max", "100", *args)

    def frame(x):
        rig.frame_id += 1
        suffix = "" if quality is None else f" OBS {quality}"
        rig.send(f"F {rig.frame_id} {time.monotonic_ns()} 1 7 {x - 20} 200 40 80 0.9{suffix}")

    try:
        assert rig.serial(1.0) == bridge.STOP
        rig.send("P")
        frame(320)
        rig.send("A 7")
        assert rig.wait_line("R A") == "R A OK"
        for x in (338, 375):
            time.sleep(0.045)
            frame(x)
        start = time.monotonic()
        first = rig.serial(0.040)
        assert first == {"T": 141, "X": 1, "Y": 0, "SPD": 1}, first
        time.sleep(max(0.0, start + 0.045 - time.monotonic()))
        frame(397)
        deadline = start + 0.085
        speeds = [first["SPD"]]
        while (remaining := deadline - time.monotonic()) > 0:
            command = rig.serial(remaining)
            if command is None:
                break
            assert command["T"] == 141 and command["X"] == 1 and command["Y"] == 0, command
            speeds.append(command["SPD"])
        return speeds
    finally:
        rig.close()


if __name__ == "__main__":
    check_option_bounds()
    for option, quality, expected in ((None, 1, False), (7, None, False), (7, 0, False), (7, 1, True)):
        speeds = path(option, quality)
        assert (7 in speeds) is expected, (option, quality, speeds)
        print(f"PASS option={option}, OBS={quality}, speeds={speeds}")
