"""List the operators of the CPU sub-graphs inside a .dxnn (sections data/cpu_models/cpu_N are ONNX bytes).
Usage: python dxnn_cpu_ops.py <file.dxnn>"""
import json
import struct
import sys

import onnx

p = sys.argv[1]
b = open(p, 'rb').read()
hdr = b[:8192]
# header: the section table is JSON-ish text in the first 8 KB in DX-COM 2.4 files; find 'cpu_models' entries
txt = hdr.decode('latin1')
i = txt.find('{')
meta = None
for j in range(len(txt), i, -1):
    try:
        meta = json.loads(txt[i:j]); break
    except Exception:
        continue
secs = meta.get('data', meta) if isinstance(meta, dict) else {}


def walk(d, path=''):
    if isinstance(d, dict):
        if 'offset' in d and 'size' in d:
            yield path, d
        for k, v in d.items():
            yield from walk(v, path + '/' + k)


for path, d in walk(meta):
    if 'cpu_models' in path:
        raw = b[8192 + d['offset']: 8192 + d['offset'] + d['size']]
        m = onnx.load_from_string(raw)
        ops = {}
        for n in m.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        print(path, 'inputs', [(x.name, [dd.dim_value for dd in x.type.tensor_type.shape.dim]) for x in m.graph.input])
        print('  outputs', [x.name for x in m.graph.output])
        print('  ops', json.dumps(ops))
        print('  first nodes', [(n.op_type, n.name) for n in m.graph.node[:12]])
