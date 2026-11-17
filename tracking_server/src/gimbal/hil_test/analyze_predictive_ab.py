import csv, hashlib, json, pathlib, sys
import numpy as np
p = pathlib.Path(sys.argv[1])
run = json.loads((p/'result.json').read_text())
if not run.get('success') or not run.get('end_view_checked') or run.get('stop_ever_failed'):
    raise SystemExit('Incomplete/failed comparison: preserve evidence, do not rank candidates')
frames = list(csv.DictReader((p/'frames.csv').open()))
serial = [json.loads(x) for x in (p/'serial.jsonl').read_text().splitlines()]
trace = [json.loads(x) for x in (p/'bridge_err.log').read_text().splitlines() if x.startswith('{"event":"control_trace"')]
protocol = [json.loads(x) for x in (p/'protocol.jsonl').read_text().splitlines()]
summary = dict(run=run, blocks={})
for name in sorted(set(r['name'] for r in frames if r['phase']=='run')):
    rows = [r for r in frames if r['name']==name and r['phase']=='run']
    ns = np.array([int(r['ns']) for r in rows])
    errors = np.array([[float(r['ximg'])-320,float(r['yimg'])-240] for r in rows])
    scene = np.array([[float(r['sx']),float(r['sy'])] for r in rows])
    dt = np.diff(ns)/1e9
    loss = np.array([r['vis']!='1' or r['lost']!='0' for r in rows])
    changes=[]; last=(0,0,0); reversals=0; prev_direction=(0,0); starts=0
    for e in serial:
        if not ns[0] <= e.get('ns',0) <= ns[-1]+40000000 or e.get('blocked') or e.get('werr'): continue
        c=e.get('cmd',{})
        if c.get('T')==141: current=(int(c.get('X',0)),int(c.get('Y',0)),int(c.get('SPD',0)))
        elif c.get('T')==135 or e.get('estop'): current=(0,0,0)
        else: continue
        if current!=last:
            changes.append(current)
            if current[2] and not last[2]: starts+=1
            if current[2]:
                for j in (0,1):
                    if current[j] and prev_direction[j] and current[j]!=prev_direction[j]: reversals+=1
                prev_direction=tuple(current[j] or prev_direction[j] for j in (0,1))
            last=current
    ids={int(r['fid']) for r in rows}
    used=[r for r in trace if r['frame_id'] in ids]
    summary['blocks'][name]=dict(frames=len(rows),duration_s=(ns[-1]-ns[0])/1e9,
        center_abs_median_px=np.median(abs(errors),axis=0).tolist(),center_abs_p95_px=np.percentile(abs(errors),95,axis=0).tolist(),
        image_path_px=float(np.linalg.norm(np.diff(scene,axis=0),axis=1).sum()),
        image_speed_p95_px_s=float(np.percentile(np.linalg.norm(np.diff(scene,axis=0),axis=1)/dt,95)),
        start_scene_px=scene[0].tolist(),end_scene_px=scene[-1].tolist(),
        loss_frames=int(loss.sum()),loss_time_fraction=float(np.sum(dt*loss[:-1])/dt.sum()),
        command_changes=len(changes),direction_reversals=reversals,motion_starts=starts,
        max_serial_speed=max([x[2] for x in changes] or [0]),controller_frames=len(used),
        forecast_used_frames=sum(any(r['forecast_used'].values()) for r in used),
        max_registration_disagreement_px=max([r['disagreement_px'] for r in protocol if r.get('event')=='image_check' and r['fid'] in ids] or [0]))
summary['raw16_files']=len(list((p/'raw16').glob('*.png')))
summary['hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file() and f.name != 'analysis.json'}
(p/'analysis.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['blocks'],indent=2))
