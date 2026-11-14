#!/usr/bin/env python3
"""Plan or execute Lab 3. Planning-only mode needs just the standard library."""

import argparse
import json
import math
import os
from pathlib import Path
import time

from simulator.patterns import PATTERNS
from simulator.planner import plan_pattern
from simulator.scene import ROOT, load_config, grid_xy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("plan","sim"),default="plan")
    parser.add_argument("--patterns",nargs="+",choices=PATTERNS,default=list(PATTERNS))
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--no-record",action="store_true")
    parser.add_argument("--gui",action="store_true")
    parser.add_argument("--speed",type=float,default=1.,help="GUI simulation/wall-time ratio: 1=real time, 5=5x, 0=unlimited (default: 1)")
    parser.add_argument("--viewer-fps",type=float,default=30.,help="Maximum GUI synchronization rate (default: 30)")
    parser.add_argument("--final-reset",action="store_true")
    parser.add_argument("--geometry-only",action="store_true",help="Probe grasp collision geometry without executing task motions")
    args=parser.parse_args()
    if not math.isfinite(args.speed) or args.speed < 0:
        parser.error("--speed must be finite and nonnegative; 0 means unlimited")
    if not math.isfinite(args.viewer_fps) or args.viewer_fps <= 0:
        parser.error("--viewer-fps must be finite and positive")
    config=load_config(args.config)
    plans={}
    for name in args.patterns:
        plan,visited=plan_pattern(PATTERNS[name])
        plans[name]=plan
        print(f"{name}: {'MODEL_INFEASIBLE' if plan is None else str(len(plan))+' placements'}; {visited} states visited",flush=True)
        if plan is not None:
            for move in plan:
                print(f"  {move.color} -> ({move.cell//3+1},{move.cell%3+1}) [{move.orientation}]")
    if args.mode=="plan":
        return
    os.environ.setdefault("MUJOCO_GL","osmesa")
    from simulator.backend import Simulation
    args.output.mkdir(parents=True,exist_ok=True)
    report={"status":"RUNNING","patterns":[],"resets":[],"config":config,
            "run_kind":"geometry" if args.geometry_only else "task_sequence",
            "physics":"MuJoCo free bodies and frictional finger contacts; no welds or block teleportation",
            "grasp_scope":"Two orthogonal top grasps, checked against finger collision meshes; other yaw angles are not searched",
            "unexpected_contact_tolerance_m":0.0008,"placement_tolerance_m":0.004}
    started=time.monotonic()
    sim=None
    try:
        sim=Simulation(config,args.output,record=not args.no_record,gui=args.gui,
                       speed=args.speed,viewer_fps=args.viewer_fps)
        report["grasp_geometry"]=sim.validate_grasp_model()
        if args.geometry_only:
            report["status"]="PASS"
            return
        locations={name:None for name in sim.blocks}
        previous=[]
        for name,plan in plans.items():
            if plan is None:
                print(f"SKIP {name}: exhaustive search failed under the two-direction grasp model",flush=True)
                report["patterns"].append({"name":name,"status":"MODEL_INFEASIBLE",
                    "reason":"Exhaustive discrete search under the validated NS/EW grasp model; no execution attempted",
                    "positions":{block:pos.tolist() for block,pos in sim.positions().items()}})
                continue
            if previous:
                print("Autonomous reset",flush=True)
                sim.recorder.label="Autonomous reset"
                for block,cell,axis in reversed(previous):
                    sim.transfer(block,sim.home_slots[block],axis)
                    locations[block]=None
                reset_positions=sim.check_arrangement(("...","...","..."),locations)
                report["resets"].append({"before_pattern":name,"status":"PASS","positions":reset_positions})
                previous=[]
            print(f"Building {name}",flush=True)
            sim.recorder.label=f"Building {name}"
            for move in plan:
                block=next(block for block in sorted(locations) if block[0]==move.color and locations[block] is None)
                sim.transfer(block,grid_xy(config,move.cell),move.orientation)
                locations[block]=move.cell
                previous.append((block,move.cell,move.orientation))
            sim.snapshot(name,f"Completed: {name}")
            actual=sim.check_arrangement(PATTERNS[name],locations)
            report["patterns"].append({"name":name,"status":"PASS","positions":actual})
            print(f"PASS {name}",flush=True)
        if args.final_reset and previous:
            print("Final autonomous reset",flush=True)
            sim.recorder.label="Final autonomous reset"
            for block,cell,axis in reversed(previous):
                sim.transfer(block,sim.home_slots[block],axis)
                locations[block]=None
            reset_positions=sim.check_arrangement(("...","...","..."),locations)
            report["resets"].append({"before_pattern":None,"status":"PASS","positions":reset_positions})
            sim.snapshot("final_reset","Completed: final autonomous reset")
            print("PASS final autonomous reset",flush=True)
        report["status"]="PASS"
        print(f"PASS simulation: {len(sim.transfers)} verified transfers, {len(report['resets'])} verified resets",flush=True)
    except Exception as error:
        report["status"]="FAIL"
        report["error"]=str(error)
        if sim:
            sim.recorder.frame(sim.data,force=True,snapshot="failure")
        raise
    finally:
        report["wall_seconds"]=time.monotonic()-started
        if sim:
            report.update(simulation_seconds=float(sim.data.time),transfers=sim.transfers,
                          steps=sim.steps,worst_joint_tracking_error_rad=sim.worst_tracking,
                          max_contact_penetration_m=sim.allowed_contact_penetration,
                          final_positions={name:pos.tolist() for name,pos in sim.positions().items()})
            sim.close()
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
