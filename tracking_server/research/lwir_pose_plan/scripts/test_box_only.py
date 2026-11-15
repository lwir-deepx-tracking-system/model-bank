"""11-15: box-only people (fewer than 2 usable keypoints, or a failed template fit) get a body box by the measured ratio in the training and
validation transforms (pose_partial.BOX_ONLY); every other caller keeps the old behaviour. Run: python test_box_only.py (from this folder)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose_partial as PP

# template: $TP, else next to this file (model bank scripts/), else the project's stage1/skeleton_box/
TPL = Path(__import__('os').environ.get('TP') or next((str(q) for q in (HERE / 'template_points.npy', HERE.parent / 'stage1' / 'skeleton_box' / 'template_points.npy') if q.is_file()), 'template_points.npy'))

PP.set_body_box(TPL)
box = np.array([[100.0, 50.0, 140.0, 150.0]], np.float32)            # a 40 x 100 px person with no keypoints
none = np.zeros((1, 17, 3), np.float32)
ign = none.copy(); ign[0, :, 2] = PP.IGNORE                              # every slot "not labelled by this data set"
one = none.copy(); one[0, 0] = [120, 55, 2]                              # one keypoint only

exp = PP.box_only_body(box[0])
assert np.allclose(exp, [120 - 0.006 * 40 - 0.5 * 0.681 * 40, 50 + 0.002 * 100, 120 - 0.006 * 40 + 0.5 * 0.681 * 40, 50 + 0.002 * 100 + 0.531 * 100], atol=1e-3), exp
for k in (none, ign, one):
    assert np.allclose(PP.to_body_boxes(box, k, (640, 480), convert_box_only=True)[0], exp)      # training / validation transforms
    assert np.allclose(PP.to_body_boxes(box, k, (640, 480))[0], box[0])                           # other callers: unchanged
PP.BOX_ONLY['on'] = False
assert np.allclose(PP.to_body_boxes(box, none, (640, 480), convert_box_only=True)[0], box[0])     # off = the runs before 11-15
PP.BOX_ONLY['on'] = True
# a person with all keypoints is fitted by the template, the same with or without the switch
P = np.load(TPL)   # (17, 2) template skeleton, y up
k = np.zeros((1, 17, 3), np.float32); k[0, :, 0] = 300 + 100 * P[:, 0]; k[0, :, 1] = 200 - 100 * P[:, 1]; k[0, :, 2] = 2
a = PP.to_body_boxes(box, k, (640, 480), convert_box_only=True)[0]; b = PP.to_body_boxes(box, k, (640, 480))[0]
assert np.allclose(a, b) and not np.allclose(a, box[0]), (a, b)
# height check against the measured relation: the converted box is 0.531 of the person box and starts at its top
assert abs((exp[3] - exp[1]) / 100.0 - 0.531) < 1e-6 and abs(exp[1] - 50.2) < 1e-3
print('box-only body box tests OK')
