"""Engineering-only stack inspection; pauses the owned simulator briefly.

Never use this while measuring flight latency. Run as WSL root for ptrace.
"""
import argparse
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
matches=[]
for path in Path("/proc").iterdir():
    if not path.name.isdigit():
        continue
    try:
        command=(path/"cmdline").read_bytes().split(b"\0")[0].decode()
        if command.startswith("/mnt/d/uav-research/idea1/scenes/openfly/") and command.endswith("AirVLN-Linux-Shipping"):
            matches.append(int(path.name))
    except (FileNotFoundError,PermissionError):
        pass
if len(matches)!=1:
    raise RuntimeError(f"Expected exactly one selected-scene process, found {len(matches)}")
pid=matches[0]
with (args.output/"threads.txt").open("w") as log:
    subprocess.run(["ps","-L","-p",str(pid),"-o","pid,tid,pcpu,comm,wchan:28"],stdout=log,stderr=log,timeout=5)
with (args.output/"thread-stacks.txt").open("w") as log:
    result=subprocess.run(["gdb","-batch","-ex","set pagination off","-ex","thread apply all bt 6","-p",str(pid)],stdout=log,stderr=log,timeout=30)
print("Inspected selected-scene process",pid,"gdb exit",result.returncode)
