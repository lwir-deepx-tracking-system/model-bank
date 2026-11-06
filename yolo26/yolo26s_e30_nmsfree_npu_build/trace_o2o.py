"""Traces the one-to-one (NMS-free) raw head of the team's YOLO26s_finetuned source, the same recipe as claude_yolo26s_01/trace_s_o2o.py with another checkpoint. Checkpoint: lwir_best_ft_onnx_handoff_20261014/best_full_ft_epoch29.pt, whose DX-COM build is byte-identical to the team's models/yolo26s_finetuned.dxnn."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, 'D:/LIG-Runs/flir_yolo26/yolo26m_m1_wholegraph_20261022_v1/claude_yolo26s_01')
import stage_s as S  # noqa: E402
L = S.L; torch = L.torch; api = L.api
CK = Path('D:/LIG-Runs/flir_yolo26/lwir_best_ft_onnx_handoff_20261014/best_full_ft_epoch29.pt'); SHA = '385621f65165f181421e1b4e4a209e32b7cb08d5f8258e95c54866439e1e2329'
m = api.load_source(CK, SHA).float().cpu().eval().requires_grad_(False); m.model[-1].end2end = False; m.criterion = None
assert getattr(m.model[-1], 'one2one_cv2', None) is not None, 'no one2one head'
print('classes', m.nc if hasattr(m, 'nc') else None, 'head', type(m.model[-1]).__name__)
b = L.RawBranchO2O(m).eval()
with torch.inference_mode():
    x = torch.rand(1, 3, 640, 640); t = torch.jit.trace(S.RawTuple(b).eval(), x, check_trace=False, strict=True)
    p = HERE / 'S_o2o.pt'; torch.jit.save(t, str(p)); ld = torch.jit.load(str(p)).eval()
    y = torch.rand(1, 3, 640, 640); assert all(torch.allclose(u, v, atol=1e-4) for u, v in zip(S.RawTuple(b)(y), ld(y)))
print('o2o trace ok', L.sha(p))
