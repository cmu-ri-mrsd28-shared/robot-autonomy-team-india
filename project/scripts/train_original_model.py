import json, time, hashlib
from pathlib import Path
from argparse import Namespace
import torch
from catchbox.learning.train import run_task,evaluate
from catchbox.learning.environment import CatchCore
from catchbox.learning.report import video,curve

root=Path('results/crossq-original-model');out=root/'box'
if out.exists() and (any((out/n).exists() for n in ['spec.json','training.jsonl','checkpoint.zip','final_model.zip'])):
    raise RuntimeError('Existing run artifacts; refusing to restart or overwrite')
root.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.set_num_interop_threads(1)
args=Namespace(seed=112021,steps=300000,eval_episodes=50,prior_episodes=64,threads=4,printed_box=True,validation_seed=20261202,heldout_seed=20261203)
print('Original model initialization; no previous model or replay loaded',flush=True)
summary=run_task('box',out,args)
bank=json.loads((out/'heldout-throws.json').read_text())
# Generate representative baseline clips from the same already-scored throws.
baseline=evaluate(None,CatchCore('box',printed_box=True),bank,out/'scripted-examples')
(root/'summary.json').write_text(json.dumps([summary,{'tool':'paddle','status':'not_started','reason':'Not included'},{'tool':'gripper','status':'not_started','reason':'Not included'}],indent=2))
parts=[]
for label,folder in [('Original model','examples'),('Scripted baseline','scripted-examples')]:
 for tag in ['passing','failing']:
  src=out/folder/f'{tag}.json'; dest=root/'media'/f'{folder}-{tag}.mp4'
  if src.exists():
   record=json.loads(src.read_text());record['controller']=label;src.write_text(json.dumps(record));video('box',src,dest,printed_box=True)
   parts.append(f'<article><h3>{label} · {tag}</h3><video controls preload="metadata" poster="media/{dest.stem}.jpg" src="media/{dest.name}"></video></article>')
  else:parts.append(f'<article><h3>{label} · {tag}</h3><p>No example observed.</p></article>')
validation=json.loads((out/'validation.json').read_text());fig=curve([(r['step'],100*r['rate']) for r in validation],'Validation retention','Percent')
result=json.loads((out/'evaluation.json').read_text());entries=sum(r['entered'] for r in result['results']);base_entries=sum(r['entered'] for r in baseline['results'])
assert result['throws_sha256']==baseline['throws_sha256']
html=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Original model printed box</title><style>body{{font:16px system-ui;max-width:1000px;margin:30px auto;padding:0 20px;color:#183039}}table{{border-collapse:collapse}}td,th{{padding:9px 20px;border-bottom:1px solid #ddd;text-align:left}}.videos{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}video{{width:100%}}svg{{max-width:600px}}@media(max-width:650px){{.videos{{grid-template-columns:1fr}}}}</style></head><body><h1>Original model · printed box</h1><p>Simulation only · 15 × 15 × 10 cm outer box · 300,000 training transitions</p><table><tr><th>Controller</th><th>Entered</th><th>Retained for one second</th></tr><tr><td>Original model</td><td>{entries}/50</td><td>{result['successes']}/50</td></tr><tr><td>Scripted baseline</td><td>{base_entries}/50</td><td>{baseline['successes']}/50</td></tr></table><p>Both use the same 50 frozen held-out throws. Retention allows 1 mm side and top tolerance for numerical contact jitter; final checkpoint alone determines the score.</p><h2>Examples</h2><div class="videos">{''.join(parts)}</div><h2>Diagnostic validation</h2>{fig}<p>Policy and critics initialized from scratch, with 64 scripted prior episodes followed by one uninterrupted 300,000-step run. </p><p>The mesh is rendered from the supplied STL. Five rectangular collision pieces approximate the rounded cavity; the simulated gripper holds the handle rigidly without slip. STL units were confirmed as millimeters; mass is assumed to be 250 g. Estimated ball state is exact simulation data.</p></body></html>'''
(root/'report.html').write_text(html);print('Report complete',flush=True)
