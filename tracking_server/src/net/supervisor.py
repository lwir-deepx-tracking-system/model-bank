"""Internet / Tailscale supervisor: decides 'online' or 'lan-only' and publishes a small state file.

The decision only informs (page note, logs, decisions of other tools). The server itself never needs the internet:
it listens on the Tailscale address (when present) and on the direct-cable addresses at the same time.

Rules: lan-only after FAIL_LIMIT failed checks in a row (5 s apart = 15 s), online again after OK_LIMIT good checks in a row.
"""
import json
import os
import time

from .netutil import run, tcp_ok

INTERNET_TARGETS = (('1.1.1.1', 443), ('8.8.8.8', 53), ('9.9.9.9', 443))


def probe_internet(targets=INTERNET_TARGETS, timeout=1.5, connect=tcp_ok):
    """True when any well-known public address accepts a TCP connection (no DNS needed, no data is sent)."""
    return any(connect(host, port, timeout) for host, port in targets)


def probe_tailscale(ip_command='ip', tailscale_command='tailscale'):
    """'running' / 'down' / 'absent' from the tailscale0 address and, best effort, `tailscale status --json`."""
    if not run([ip_command, '-4', '-o', 'addr', 'show', 'dev', 'tailscale0']).strip():
        return 'absent'
    out = run([tailscale_command, 'status', '--json'], timeout=4.0)
    if not out:
        return 'running'   # address present, CLI not answering: do not claim it is down
    try:
        return 'running' if json.loads(out).get('BackendState') == 'Running' else 'down'
    except ValueError:
        return 'running'


def write_state(path, state):
    """Atomic, world-readable state file (the web server reads it)."""
    tmp = '%s.%d.tmp' % (path, os.getpid())
    with open(tmp, 'w') as handle:
        json.dump(state, handle, sort_keys=True)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


class Supervisor:
    def __init__(self, probe=probe_internet, tailscale=probe_tailscale, fail_limit=3, ok_limit=2, clock=time.time):
        self.probe, self.tailscale = probe, tailscale
        self.fail_limit, self.ok_limit, self.clock = fail_limit, ok_limit, clock
        self.mode = None            # None until the first check
        self.internet = None        # raw result of the last check
        self.tailscale_state = 'absent'
        self.fails = self.oks = 0
        self.since = clock()
        self.changes = []           # (epoch, mode) log of the transitions, for the log file

    def tick(self, check_tailscale=True):
        ok = bool(self.probe())
        self.internet = ok
        if check_tailscale:
            self.tailscale_state = self.tailscale()
        previous = self.mode
        if previous is None:
            self._switch('online' if ok else 'lan-only')
        elif ok:
            self.fails, self.oks = 0, self.oks + 1
            if previous == 'lan-only' and self.oks >= self.ok_limit:
                self._switch('online')
        else:
            self.oks, self.fails = 0, self.fails + 1
            if previous == 'online' and self.fails >= self.fail_limit:
                self._switch('lan-only')
        return self.mode != previous

    def _switch(self, mode):
        self.mode, self.since = mode, self.clock()
        self.fails = self.oks = 0
        self.changes.append((self.since, mode))

    def state(self):
        return dict(mode=self.mode, internet=self.internet, tailscale=self.tailscale_state, since=round(self.since))
