#!/usr/bin/env python3
"""Writes the gimbal's STOP command ({"T":135}) three times straight to its serial device. Standard library only.
Called by run_v27_hil.sh right after the test program has ended, whatever way it ended (also after a crash or a kill), before anything else.
A STOP to a gimbal that stands still does nothing (the bridge sends the same command every time it starts).
Devices tried in this order until one takes a complete STOP: the one given as the argument, the gimbal's fixed name under /dev/serial/by-id,
/dev/ttyUSB0. Exit code 0 only when at least one STOP was written completely; the line printed says to which device, or that none was written."""
import glob, os, sys, termios, time, tty

MSG = b'{"T":135}\n'
c = ([sys.argv[1]] if len(sys.argv) > 1 else []) + sorted(glob.glob('/dev/serial/by-id/usb-Silicon_Labs_CP2102N_*')) + ['/dev/ttyUSB0']; errs = []
for dev in dict.fromkeys(c):
    try:
        fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK); n = 0
        try:
            tty.setraw(fd); a = termios.tcgetattr(fd)
            a[4] = a[5] = termios.B115200; a[2] |= termios.CLOCAL | termios.CREAD; a[2] &= ~(termios.PARENB | termios.CSTOPB | termios.CRTSCTS)
            termios.tcsetattr(fd, termios.TCSANOW, a)
            for _ in range(3):
                try: n += int(os.write(fd, MSG) == len(MSG))
                except OSError as e: errs.append('%s: %r' % (dev, e))
                time.sleep(0.05)
        finally: os.close(fd)
        if n: print('STOP written %d times to %s' % (n, dev)); sys.exit(0)
    except Exception as e: errs.append('%s: %r' % (dev, e))
print('STOP NOT WRITTEN: %s' % ' | '.join(errs)); sys.exit(1)
