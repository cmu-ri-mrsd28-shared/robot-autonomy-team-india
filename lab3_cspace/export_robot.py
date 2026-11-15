"""Generate joint plans and export a standalone robot-only folder offline."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab3_cspace.robot_plans import generate


def package_planner(destination):
    """Copy only NumPy planning modules; never copy simulation/controller code."""
    source=Path(__file__).resolve().parent
    package=Path(destination)/'planner'; package.mkdir(parents=True,exist_ok=True)
    (package/'__init__.py').write_text('"""Generated NumPy-only planning package. Source: lab3_cspace."""\n')
    for name in ('kinematics','task','grasp','motion','collision','plan','robot_plans'):
        text=(source/(name+'.py')).read_text()
        if name=='plan':
            tree=ast.parse(text)
            omitted=set()
            for node in tree.body:
                if isinstance(node,ast.If) or (isinstance(node,ast.FunctionDef) and node.name=='main'):
                    omitted.update(range(node.lineno,node.end_lineno+1))
            text=''.join(line for i,line in enumerate(text.splitlines(keepends=True),1) if i not in omitted)
        text=text.replace('from lab3_cspace.','from .')
        (package/(name+'.py')).write_text('# Generated from lab3_cspace/'+name+'.py by export_robot.py.\n'+text)
    shutil.copy2(source/'panda_boxes.json',package/'panda_boxes.json')
    shutil.copy2(source.parents[0]/'simulation/assets/LICENSE',package/'LICENSE_PANDA')
    (package/'NOTICE_PANDA').write_text('Panda collision vertices derived from simulation/assets/meshes in this repository.\nThe original mesh data are Apache-2.0 licensed; see LICENSE_PANDA.\nData converted to JSON with enclosing boxes; geometry is not changed.\n')


def main():
    source=Path(__file__).resolve().parents[1]/'lab3_cspace_robot'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(source/'config.json'))
    parser.add_argument('--start-state',help='State JSON captured by robot run.py --state')
    parser.add_argument('--output',default=str(source))
    parser.add_argument('--runtime-only',action='store_true',help='Package runtime/planner without regenerating plans')
    args=parser.parse_args()
    start=None if not args.start_state else json.loads(Path(args.start_state).read_text())['q']
    dest=Path(args.output).resolve(); dest.mkdir(parents=True,exist_ok=True)
    if dest!=source:
        for name in ('run.py','generate_plan.py','joint_executor.py','panda_model.py','requirements.txt','README.md'):
            shutil.copy2(source/name,dest/name)
    package_planner(dest)
    if not args.runtime_only: generate(args.config,dest,start)
    print('Exported robot runtime and NumPy planner to',dest)


if __name__=='__main__': main()
