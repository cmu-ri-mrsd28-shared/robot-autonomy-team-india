"""Concise two-page offline HTML report, backed only by recorded run artifacts."""
import html,json,subprocess
from pathlib import Path
import numpy as np
from catchbox.core import Config
from catchbox.sim import Simulation
from catchbox.experiments import setup
from catchbox.video import Renderer


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default

def video(tool,source,destination,printed_box=False):
    record=read(source);cfg=Config(duration=3.5,control_dt=.02,box_width=.12 if printed_box else .10 if tool=='box' else .05,box_depth=.12 if printed_box else .10 if tool=='box' else .05,box_height=.09 if printed_box else .05 if tool=='box' else .12,printed_box=printed_box)
    sim=Simulation(cfg) if tool=='box' else setup(tool,cfg)
    renderer=Renderer(sim);renderer.main_lookat=[.85,0,.6];renderer.main_scale=260
    destination.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x600',
                           '-r','25','-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p',
                           '-movflags','+faststart',str(destination)],stdin=subprocess.PIPE)
    try:
        for i,row in enumerate(record['trace']):
            frame=renderer.frame(row,f'{tool.upper()} | {record.get("controller", "trained CrossQ")} | held-out example',record['result']['outcome'])
            proc.stdin.write(frame.tobytes())
            if i==min(20,len(record['trace'])-1):frame.save(destination.with_suffix('.jpg'))
    finally:proc.stdin.close()
    if proc.wait()!=0:raise RuntimeError('Video encoding failed')
    subprocess.run(['ffmpeg','-v','error','-i',str(destination),'-f','null','-'],check=True)


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def curve(values,title,ylabel):
    values=[(float(x),float(y)) for x,y in values if np.isfinite(x) and np.isfinite(y)]
    if not values:return '<p class="muted">No recorded '+html.escape(title.lower())+'.</p>'
    xmin=0;xmax=max(x for x,y in values) or 1;ymin=min(0,min(y for x,y in values));ymax=max(y for x,y in values)
    ymax=max(ymax,ymin+1e-6)
    pts=' '.join(f'{55+510*(x-xmin)/(xmax-xmin):.1f},{175-140*(y-ymin)/(ymax-ymin):.1f}' for x,y in values)
    return f'''<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 600 220" role="img" aria-label="{html.escape(title)}"><path d="M55 30V175H575" fill="none" stroke="#8c97a5"/><polyline points="{pts}" fill="none" stroke="#206e73" stroke-width="2"/><text x="6" y="32">{ymax:.2g}</text><text x="6" y="178">{ymin:.2g}</text><text x="55" y="200">0</text><text x="520" y="200">{xmax/1000:.0f}k</text><text x="240" y="218">Training transitions</text><text x="65" y="23">{html.escape(ylabel)}</text></svg></figure>'''


def build_report(root,render=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    summaries=read(root/'summary.json',[]);bytool={r['tool']:r for r in summaries}
    rows=[];videos=[];details=[];specs=[]
    for tool in ['box','paddle','gripper']:
        summary=bytool.get(tool,dict(status='not_started',reason='Awaiting execution'))
        spec=read(root/tool/'spec.json',{});ev=read(root/tool/'evaluation.json',{})
        bank=read(root/tool/'heldout-throws.json',{});train_path=root/tool/'training.jsonl'
        logs=[json.loads(x) for x in train_path.read_text().splitlines()] if train_path.exists() else []
        status=summary['status'];trained=status=='completed'
        score=f"{summary['successes']}/{summary['episodes']} ({100*summary['rate']:.1f}%)" if trained else '—'
        baseline=f"{100*summary['baseline_rate']:.1f}%" if trained else '—'
        movement=f"{100*np.median([r['arm_movement_m'] for r in ev['results']]):.1f} cm" if ev else '—'
        rows.append([tool.title(),score,baseline,movement,'Pass' if trained and summary['gate_pass'] else 'Below 80%' if trained else html.escape(summary.get('reason',status))])
        parts=[]
        for tag in ['passing','failing']:
            src=root/tool/'examples'/f'{tag}.json';dest=root/'media'/f'{tool}-{tag}.mp4'
            baseline_clip=False
            if tag=='passing' and trained and not src.exists():
                baseline=read(root/tool/'baseline.json',{})
                known=next((r for r in baseline.get('results',[]) if r['success']),None)
                if known is not None:
                    # Replay a known, already scored baseline success; never relabel it learned.
                    from .environment import CatchCore
                    from catchbox.core import Throw
                    core=CatchCore(tool,record=True);core.reset(Throw(**known['throw']),known['meta'])
                    while True:
                        _,_,term,trunc,info=core.step(np.zeros(7))
                        if term or trunc:break
                    if not info['result']['success']:raise RuntimeError('Baseline replay did not reproduce its success')
                    src=root/tool/'baseline-passing.json'
                    src.write_text(json.dumps(dict(controller='scripted baseline',result=info['result'],trace=core.trace)))
                    baseline_clip=True
            label=tag.title()+(' · scripted baseline' if baseline_clip else ' · trained CrossQ')
            if src.exists():
                if render and not dest.exists():video(tool,src,dest)
                if dest.exists():parts.append(f'<div><b>{label}</b><video controls preload="metadata" poster="media/{tool}-{tag}.jpg" src="media/{tool}-{tag}.mp4"></video></div>')
                else:parts.append(f'<p>{tag.title()}: trajectory recorded; video pending.</p>')
            else:parts.append(f'<p class="muted">{tag.title()}: '+('none observed in this evaluation.' if trained else 'no trained rollout; task not run.')+'</p>')
        videos.append(f'<article><h3>{tool.title()}</h3><div class="pair">'+''.join(parts)+'</div></article>')
        if spec:
            specs.append([tool.title(),f"{spec['parameters']['actor']:,}",f"{spec['parameters']['critics']:,}",f"{summary.get('online_steps',0):,}",f"{summary.get('prior_transitions',0):,}",f"{summary.get('train_seconds',0)/60:.1f} min"])
        if ev:
            bins=[]
            for label,key,edges in [('Distance','distance_m',[.5,.83,1.17,1.51]),('Angle','angle_deg',[-81,-40,0,40,81]),('Speed','speed_m_s',[0,3,4,5.01])]:
                for lo,hi in zip(edges[:-1],edges[1:]):
                    rs=[r for r in ev['results'] if lo<=r['meta'][key]<hi]
                    if rs:bins.append([label,f'{lo:g}–{hi:g}',len(rs),f"{100*np.mean([r['success'] for r in rs]):.1f}%"])
            outcomes={o:sum(r['outcome']==o for r in ev['results']) for o in sorted({r['outcome'] for r in ev['results']})}
            ci=ev['wilson95'];rejects=len(bank.get('rejections',[]));proposed=bank.get('proposed',0)
            detail=f'<h3>{tool.title()}</h3><p>95% Wilson interval: {100*ci[0]:.1f}–{100*ci[1]:.1f}%. Launch filtering: {rejects}/{proposed} proposals rejected before evaluation.</p>'
            detail+=table(['Stratum','Range (m / ° / m·s⁻¹)','Throws','Retained'],bins)
            detail+='<p>Outcomes: '+html.escape(', '.join(f'{k}: {v}' for k,v in outcomes.items()))+'.</p>'
            val=read(root/tool/'validation.json',[])
            detail+='<div class="pair">'+curve([(r['step'],100*r['rate']) for r in val],'Validation retention','Percent')+curve([(r['step'],r['train/critic_loss']) for r in logs if 'train/critic_loss' in r],'Critic loss','MSE')+'</div>'
            details.append(detail)
    stats=table(['Task','Actor parameters','Critic parameters','Online steps','Prior transitions','Training + validation'],specs) if specs else '<p class="muted">No model has been trained. Parameter counts and losses are unavailable.</p>'
    warning=read(root/'blocked.json',{})
    note=f'<p class="notice">{html.escape(warning["reason"])}</p>' if warning else ''
    body=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Franka catching · CrossQ report</title><style>
:root{{font-family:system-ui,sans-serif;color:#172d36;background:#f4f6f6}}body{{max-width:1060px;margin:28px auto;padding:0 22px;line-height:1.45}}h1{{font-size:26px;margin-bottom:4px}}h2{{font-size:20px}}h3{{font-size:16px}}p{{margin:9px 0}}.muted{{color:#52636b}}nav{{display:flex;gap:8px;margin:20px 0}}button{{border:1px solid #becbce;background:white;padding:9px 16px;border-radius:6px;cursor:pointer}}button[aria-selected=true]{{background:#174b50;color:white}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:9px 10px;text-align:left;border-bottom:1px solid #dbe3e5}}th{{background:#e5edee}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}article{{background:white;padding:12px 16px;margin:12px 0;border-radius:8px}}video{{width:100%;margin-top:5px;background:#15202b;max-height:280px}}figure{{margin:12px 0}}figcaption{{font-weight:600}}svg text{{font-size:12px;fill:#465a63}}.notice{{padding:12px;background:#fff1ce}}details{{margin-top:14px}}footer{{font-size:12px;color:#52636b;margin-top:20px}}[hidden]{{display:none!important}}@media(max-width:650px){{.pair{{grid-template-columns:1fr}}}}@media print{{nav{{display:none}}section[hidden]{{display:block!important}}#breakdown{{break-before:page}}video{{max-height:140px}}}}
</style></head><body><h1>Franka catching · CrossQ</h1><p class="muted">Simulation-only · 10 × 10 × 5 cm box · 15 cm paddle · stock gripper</p>{note}<nav role="tablist"><button id="tab-overview" role="tab" aria-controls="overview" aria-selected="true">1 · Results & training</button><button id="tab-breakdown" role="tab" aria-controls="breakdown" aria-selected="false">2 · Breakdown</button></nav>
<section id="overview" role="tabpanel" aria-labelledby="tab-overview"><h2>Held-out results</h2>{table(['Task','CrossQ retention','Scripted baseline','Median arm movement','80% gate'],rows)}<p class="muted">Success = continuous one-second retention. Gripper runs only if both earlier tasks pass. Final checkpoint evaluated once on frozen held-out throws; no held-out checkpoint selection.</p>
<h2>Passing and failing rollouts</h2><p class="muted">First observed trained example of each outcome. If the learned policy has no successes, a passing scripted baseline is explicitly labeled instead. Half-speed playback; CPU rendering of recorded MuJoCo states.</p>{''.join(videos)}
<h2>Training</h2>{stats}<p><b>Algorithm:</b> CrossQ residual control, maintained sb3-contrib implementation. Actor 128–128; twin critics 256–256; batch renormalization, no target critic. Position/velocity/acceleration, robot state and trajectory prediction form a 60-value observation. Seven bounded motion/retention corrections at 50 Hz; torque control at 1 kHz.</p>
<p><b>Data & optimization:</b> 64 scripted prior episodes including correctly labeled failures, then one CPU run per evaluated task (step counts above). Replay capacity 250k, batch 256, one critic update per transition, actor every third update. Adam, learning rate 0.001, discount 0.99, learned entropy temperature initialized at 0.1. Critic loss is the entropy-adjusted Bellman MSE; actor minimizes temperature × log probability minus Q.</p>
<p><b>Curriculum:</b> one continuous run widens angle/distance through its first half, then trains on full breadth. Dense interception and retention rewards, a one-second success bonus, and action/smoothness penalties. Final evaluation never uses training or validation seeds.</p>
<p><b>Limits:</b> oracle sensing, gravity-only flight, provisional bounce/friction and rigid tool attachments. Jaw speed/delay are assumptions, not lab measurements. One seed per task; no guarantee of convergence or hardware transfer.</p></section>
<section id="breakdown" role="tabpanel" aria-labelledby="tab-breakdown" hidden><h2>Coverage, outcomes & learning</h2><p>Throws span 0.5–1.5 m distance, ±80° approach and 0.50–0.85 s flight time. Intercepts are displaced 6.5–14 cm. Speed, IK and arrival-deadline filters are applied before scoring and logged.</p>{''.join(details) or '<p>No completed evaluations or learning curves yet.</p>'}</section>
<footer>Artifacts: spec.json, training.jsonl, validation.json, heldout-throws.json, evaluation.json, baseline.json, model checkpoints and recorded examples. <a href="https://sb3-contrib.readthedocs.io/en/master/modules/crossq.html">CrossQ implementation</a> · <a href="https://proceedings.iclr.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract-Conference.html">Paper</a></footer>
<script>for(const name of ['overview','breakdown']){{document.getElementById('tab-'+name).addEventListener('click',()=>{{for(const other of ['overview','breakdown']){{document.getElementById(other).hidden=other!==name;document.getElementById('tab-'+other).setAttribute('aria-selected',String(other===name));}}}});}}</script></body></html>'''
    (root/'report.html').write_text(body)
    return root/'report.html'

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--no-render',action='store_true');a=p.parse_args()
    print(build_report(a.root,not a.no_render))
