"""One CrossQ lineage per task; box -> paddle -> conditional gripper; no search/sweeps."""
import argparse,json,time,platform,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
from catchbox.core import Throw
from .environment import CatchCore,gym_env
from .trajectory import freeze_bank


def evaluate(model,core,bank,trace_dir=None):
    results=[];examples={};start=time.perf_counter()
    for i,item in enumerate(bank['throws']):
        core.record=trace_dir is not None
        obs=core.reset(Throw(**item['throw']),item['meta'])
        while True:
            action=np.zeros(7) if model is None else model.predict(obs,deterministic=True)[0]
            obs,_,terminated,truncated,info=core.step(action)
            if terminated or truncated:break
        result=info['result'];result['episode']=i;results.append(result)
        tag='passing' if result['success'] else 'failing'
        if trace_dir is not None and tag not in examples:
            trace_dir.mkdir(parents=True,exist_ok=True)
            p=trace_dir/f'{tag}.json';p.write_text(json.dumps(dict(result=result,trace=core.trace)))
            examples[tag]=str(p)
    n=len(results);k=sum(r['success'] for r in results);p=k/n;z=1.96
    mid=(p+z*z/(2*n))/(1+z*z/n);delta=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(n=n,successes=k,rate=p,wilson95=[mid-delta,mid+delta],results=results,examples=examples,
                wall_s=time.perf_counter()-start,throws_sha256=bank.get('sha256'))


def run_task(tool,out,args):
    import torch
    from sb3_contrib import CrossQ
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    out.mkdir(parents=True,exist_ok=True)
    if (out/'final_model.zip').exists():raise RuntimeError('Existing completed run: choose a new output folder explicitly')
    printed_box=getattr(args,'printed_box',False)
    val_seed=getattr(args,'validation_seed',20261120)
    test_seed=getattr(args,'heldout_seed',20261121)
    env=gym_env(tool,args.seed,True,args.steps,printed_box=printed_box)
    validation=CatchCore(tool,seed=val_seed,printed_box=printed_box)
    val_bank=freeze_bank(validation.sim,val_seed,32,out/'validation-throws.json')
    evaluation=CatchCore(tool,seed=test_seed,printed_box=printed_box)
    eval_bank=freeze_bank(evaluation.sim,test_seed,args.eval_episodes,out/'heldout-throws.json')
    model=CrossQ('MlpPolicy',env,learning_rate=1e-3,buffer_size=250000,learning_starts=1000,
                 batch_size=256,gamma=.99,train_freq=1,gradient_steps=1,policy_delay=3,
                 ent_coef='auto_0.1',policy_kwargs=dict(net_arch=dict(pi=[128,128],qf=[256,256]),n_critics=2),
                 seed=args.seed,device='cpu',verbose=0)
    model.set_logger(configure(str(out),['json']))
    counts=dict(actor=sum(p.numel() for p in model.actor.parameters()),
                critics=sum(p.numel() for p in model.critic.parameters()),
                total=sum(p.numel() for p in model.policy.parameters()))
    spec=dict(algorithm='sb3-contrib CrossQ',actor=[128,128],critics=[256,256],n_critics=2,
              parameters=counts,learning_rate=.001,batch_size=256,gamma=.99,gradient_steps_per_transition=1,
              actor_update_interval=3,entropy='auto_0.1',buffer_size=250000,device='cpu',threads=args.threads,
              online_steps=args.steps,seed=args.seed,physics=vars(env.core.cfg),
              observation_dimension=env.observation_space.shape[0],action_dimension=7,
              prior_episodes=args.prior_episodes,optimizer='Adam',
              optimizer_kwargs=model.actor.optimizer.defaults,
              versions={k:importlib.metadata.version(k) for k in ['torch','sb3-contrib','stable-baselines3','mujoco','robosuite']},
              platform=platform.platform(),bank_seeds=dict(validation=val_seed,heldout=test_seed))
    # Adam defaults may include tuples/bools; JSON-friendly with no model tensors.
    (out/'spec.json').write_text(json.dumps(spec,indent=2,default=str))
    print(f'{tool}: {counts}; collecting {args.prior_episodes} scripted prior episodes',flush=True)
    prior=CatchCore(tool,seed=10001,printed_box=printed_box);prior_count=0;prior_success=0;prior_results=[]
    start=time.perf_counter()
    for i in range(args.prior_episodes):
        obs=prior.reset()
        while True:
            action=np.zeros(7,dtype=np.float32);nxt,reward,term,trunc,info=prior.step(action)
            replay_info={'TimeLimit.truncated':trunc and not term}
            model.replay_buffer.add(obs[None],nxt[None],action[None],np.array([reward]),np.array([term or trunc]),[replay_info])
            prior_count+=1;obs=nxt
            if term or trunc:break
        prior_success+=int(info['result']['success']);prior_results.append(info['result'])
    (out/'prior.json').write_text(json.dumps(dict(transitions=prior_count,successes=prior_success,
                                                episodes=prior_results,wall_s=time.perf_counter()-start),indent=2))
    class Progress(BaseCallback):
        def __init__(self):super().__init__();self.history=[];self.started=time.perf_counter();self.last_time=self.started
        def _on_step(self):
            if self.num_timesteps%1000==0:
                vals={k:float(v) for k,v in self.model.logger.name_to_value.items() if isinstance(v,(int,float,np.number))}
                vals.update(step=self.num_timesteps,wall_s=time.perf_counter()-self.started,
                            curriculum=env.core.level,episodes=self.model._episode_num)
                with (out/'training.jsonl').open('a') as f:f.write(json.dumps(vals)+'\n')
                self.model.logger.dump(self.num_timesteps)
            if self.num_timesteps%10000==0:
                self.model.save(out/'checkpoint')
                ev=evaluate(self.model,validation,val_bank)
                brief={k:v for k,v in ev.items() if k!='results'};brief['step']=self.num_timesteps
                self.history.append(brief);(out/'validation.json').write_text(json.dumps(self.history,indent=2))
                print(f'{tool}: {self.num_timesteps}/{args.steps}, validation {ev["successes"]}/{ev["n"]}, elapsed {time.perf_counter()-self.started:.0f}s',flush=True)
            return True
    callback=Progress();start=time.perf_counter()
    model.learn(total_timesteps=args.steps,callback=callback,log_interval=100)
    train_seconds=time.perf_counter()-start
    model.save(out/'final_model');model.save_replay_buffer(out/'replay.pkl')
    # Final checkpoint only: held-out evaluation is not a checkpoint selection loop.
    learned=evaluate(model,evaluation,eval_bank,out/'examples')
    baseline=evaluate(None,evaluation,eval_bank)
    (out/'evaluation.json').write_text(json.dumps(learned,indent=2))
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2))
    summary=dict(tool=tool,status='completed',rate=learned['rate'],successes=learned['successes'],
                 episodes=learned['n'],baseline_rate=baseline['rate'],gate_pass=learned['rate']>=.8,
                 train_seconds=train_seconds,online_steps=model.num_timesteps,prior_transitions=prior_count,
                 parameters=counts,examples=learned['examples'],wilson95=learned['wilson95'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/crossq'))
    p.add_argument('--steps',type=int,default=150000);p.add_argument('--eval-episodes',type=int,default=200)
    p.add_argument('--prior-episodes',type=int,default=64);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=112019);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.steps<1 or a.eval_episodes<1:p.error('Positive training/evaluation budgets required')
    import torch
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    a.out.mkdir(parents=True,exist_ok=True)
    if a.smoke:
        from sb3_contrib import CrossQ
        from stable_baselines3.common.env_checker import check_env
        env=gym_env('box',a.seed,True,a.steps);check_env(env,warn=True)
        model=CrossQ('MlpPolicy',env,device='cpu',learning_starts=16,batch_size=32,
                     policy_kwargs=dict(net_arch=dict(pi=[128,128],qf=[256,256])),seed=a.seed)
        model.learn(total_timesteps=64);model.save(a.out/'smoke-model');print('CROSSQ_SMOKE_OK',flush=True);return
    summaries=[]
    for tool in ['box','paddle']:
        summaries.append(run_task(tool,a.out/tool,a))
        (a.out/'summary.json').write_text(json.dumps(summaries,indent=2))
    if all(r['gate_pass'] for r in summaries):summaries.append(run_task('gripper',a.out/'gripper',a))
    else:summaries.append(dict(tool='gripper',status='not_started',reason='Both box and paddle must reach 80% held-out success'))
    (a.out/'summary.json').write_text(json.dumps(summaries,indent=2))
    from .report import build_report
    build_report(a.out)
if __name__=='__main__':main()
