"""Speaks the team PC GUI's control protocol (TCP, magic 0x4c574952, header category/command/size) to the bridge: ModelCount, ChangeModel, TrackerCount.
Usage: python3 gui_model_client.py <host> <port> <step...>   steps: count | tcount | change:<id> | wait:<seconds>   (replies are printed as they arrive)"""
import socket, struct, sys, time
MAGIC = 0x4c574952
NAMES = {(4, 0x81): 'ModelCountReply', (4, 0x82): 'TrackerCountReply', (4, 0x83): 'ModelResult', (4, 0x84): 'TrackerResult', (1, 2): 'Init', (1, 3): 'Status', (1, 1): 'Ack', (1, 4): 'Ping'}
RES = ['Ok', 'InvalidId', 'Busy', 'CreationFailed', 'Restored', 'RestoreFailed', 'NotReady']
PHASE = {1: 'Accepted', 2: 'Completed'}
host, port, steps = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
s = socket.create_connection((host, port), timeout=5); s.settimeout(0.2); buf = b''; req = [100]; t0 = time.time()


def send(cat, cmd, payload=b''):
    s.sendall(struct.pack('>IBBH', MAGIC, cat, cmd, len(payload)) + payload)


def pump(seconds):
    global buf
    end = time.time() + seconds
    while time.time() < end:
        try:
            d = s.recv(4096)
            if not d: print('  [closed by the bridge]'); return
            buf += d
        except socket.timeout: pass
        while len(buf) >= 8:
            m, cat, cmd, n = struct.unpack('>IBBH', buf[:8])
            if m != MAGIC: print('  bad magic', hex(m)); buf = b''; return
            if len(buf) < 8 + n: break
            pl = buf[8:8 + n]; buf = buf[8 + n:]; name = NAMES.get((cat, cmd), 'cat%d cmd%d' % (cat, cmd))
            if (cat, cmd) == (4, 0x81): txt = 'req %d count %d' % struct.unpack('>II', pl)
            elif (cat, cmd) in ((4, 0x83), (4, 0x84)): r, ph, rs = struct.unpack('>IBB', pl); txt = 'req %d %s %s' % (r, PHASE.get(ph, ph), RES[rs] if rs < len(RES) else rs)
            elif (cat, cmd) == (1, 4): txt = ''; send(1, 1, bytes([1, 4, 0]))   # answer a ping with an Ack (category 1 cmd 4 result Ok)
            else: txt = pl.hex()
            if (cat, cmd) != (1, 4) and (cat, cmd) != (1, 3): print('  %6.2fs  %s %s' % (time.time() - t0, name, txt), flush=True)


pump(1.0)
for st in steps:
    print('>', st, flush=True)
    if st == 'count': req[0] += 1; send(4, 1, struct.pack('>I', req[0]))
    elif st == 'tcount': req[0] += 1; send(4, 2, struct.pack('>I', req[0]))
    elif st.startswith('change:'): req[0] += 1; send(4, 3, struct.pack('>II', req[0], int(st.split(':')[1])))
    elif st.startswith('wait:'): pump(float(st.split(':')[1])); continue
    pump(0.6)
s.close()
