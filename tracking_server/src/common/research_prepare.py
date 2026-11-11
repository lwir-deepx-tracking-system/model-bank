"""PC 에서 장치에 올릴 폴더를 만드는 기능 (연구용 옵션 --prepare)."""
import json
from pathlib import Path
from ..common.shared_values import HERE, ROOT, sha


def prepare(args):
    """PC: upload_track/ = m1_track_eval.py + models/{DRT512A,DRT512B,<vit>}.dxnn + models/MANIFEST.json."""
    import shutil
    if not args.mock_vit_src or (args.det_src == 'NEWM' and not args.mock_rt480_onnx): raise SystemExit('--prepare needs --mock-vit-src <PC folder> (and --mock-rt480-onnx <PC folder> with --det-src NEWM): these options have no default in the public repository')
    B0 = ROOT / 'claude_b0_tile_01'; VIT_SRC = Path(args.mock_vit_src)   # ViT DXNNs and SYNTH_PLAN.json stay in claude_vit_reid_01
    if args.det_src == 'NEWM':   # new m RT480 PTQ tiles (kit deploy_all, opt_level 1, model-bank candidates): run with --det rt480
        K = Path(args.mock_rt480_onnx)
        src = {'NEWMRT480A': (K / 'NEWMRT480A' / 'compiler' / 'NEWMRT480A.dxnn', '1f8e4a32436a5cab07ed67dca3b4c03d6df57fa2235056db1966323a61b09bed'),
               'NEWMRT480B': (K / 'NEWMRT480B' / 'compiler' / 'NEWMRT480B.dxnn', 'b7bd1819af0b458661de234c4139827475ee286653ba5b98c545af04996d6e9b')}
        up = HERE / 'upload_track_NEWM'
    elif args.det_src == 'DRT512':
        src = {'DRT512A': (B0 / 'DRT512A' / 'compiler' / 'DRT512A.dxnn', 'a150290aa6a1ea420859d03da6940ef7181384dd5b6a2938e95a8b9d8a36fc46'),
               'DRT512B': (B0 / 'DRT512B' / 'compiler' / 'DRT512B.dxnn', 'beafc655848d58fde638c3dde066e2aacdd8322aca651933a441683391ad4040')}
        up = HERE / 'upload_track'
    elif (ROOT / 'claude_tt75_deploy_01' / args.det_src / 'upload' / 'models').exists():   # two-tower 0.75 deploy kit (KD22, KDF, ...): --det tt75 slots TTA75/TTB75
        K = ROOT / 'claude_tt75_deploy_01' / args.det_src / 'upload' / 'models'
        src = {'TTA75': (K / (args.det_src + 'TTA75.dxnn'), None), 'TTB75': (K / (args.det_src + 'TTB75.dxnn'), None)}
        up = HERE / ('upload_track_' + args.det_src)
    else:   # same 512x384 tile geometry and I/O as DRT512A/B; only the weights (fine-tuned) and the DX-COM opt_level (1) differ
        K = ROOT / 'claude_ft_deploy_o1_01' / args.det_src / 'upload' / 'models'
        src = {'DRT512A': (K / (args.det_src + 'RT512A.dxnn'), None), 'DRT512B': (K / (args.det_src + 'RT512B.dxnn'), None)}
        up = HERE / ('upload_track_' + args.det_src)
    src[args.vit] = (VIT_SRC / args.vit / 'compiler' / (args.vit + '.dxnn'), None)
    (up / 'models').mkdir(parents=True, exist_ok=True)
    man = json.loads((up / 'models' / 'MANIFEST.json').read_text()) if (up / 'models' / 'MANIFEST.json').exists() else {}
    man['_detector'] = ('NEWM = new m RT480 PTQ NEWMRT480A/B (opt_level 1, kit claude_ft_deploy_o1_01/NEWM); run with --det rt480' if args.det_src == 'NEWM' else
                        'DRT512 = original YOLO26m-RT (opt0)' if args.det_src == 'DRT512' else
                        ('%s = two-tower 0.75 %sTTA75/%sTTB75 (claude_tt75_deploy_01) in the TTA75/TTB75 slots; run with --det tt75' % ((args.det_src,) * 3)) if 'TTA75' in src else
                        '%s = fine-tuned YOLO26m-RT %sRT512A/B (opt_level 1) in the DRT512A/B slots' % (args.det_src, args.det_src))
    for n, (p, expect) in src.items():
        s = sha(p); assert expect is None or s == expect, (n, s); shutil.copyfile(p, up / 'models' / (n + '.dxnn')); man[n] = s
    (up / 'models' / 'MANIFEST.json').write_text(json.dumps(man, indent=2)); shutil.copyfile(HERE / 'm1_track_eval.py', up / 'm1_track_eval.py')
    plan = VIT_SRC / 'tracker_ab' / 'SYNTH_PLAN.json'      # --synth-plan SYNTH_PLAN.json on the M1 (synthetic occlusion; make_synth_plan.py)
    if plan.exists(): shutil.copyfile(plan, up / 'SYNTH_PLAN.json')
    print(json.dumps(dict(man, script_sha=sha(up / 'm1_track_eval.py'), synth_plan_sha=(sha(up / 'SYNTH_PLAN.json') if plan.exists() else None)), indent=1))
