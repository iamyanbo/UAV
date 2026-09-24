"""Build a private Mesa Dozen driver; does not replace system graphics libraries."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument("--driver",choices=("dzn","lavapipe"),default="dzn")
args=parser.parse_args()
base=Path.home()/"uav-graphics-build"
source=base/"mesa-25.1.9"
build=base/("build-"+args.driver)
prefix=base/("install" if args.driver=="dzn" else "install-lavapipe")
meson=base/"tools/bin/meson"
receipts=Path("/mnt/d/uav-research/idea1/environment-receipts")/(args.driver+"-25.1.9")
receipts.mkdir(parents=True,exist_ok=True)
commands=[
 [str(meson),"setup",str(build),str(source),"--prefix="+str(prefix),"-Dbuildtype=release",
  "-Dgallium-drivers="+("" if args.driver=="dzn" else "llvmpipe"), "-Dvulkan-drivers="+("microsoft-experimental" if args.driver=="dzn" else "swrast"), "-Dplatforms=x11", "-Dglx=disabled",
  "-Degl=disabled", "-Dopengl=false", "-Dgles1=disabled", "-Dgles2=disabled", "-Dllvm="+("disabled" if args.driver=="dzn" else "enabled"),
  "-Dmicrosoft-clc=disabled", "-Dbuild-tests=false", "-Dlibunwind=disabled"],
 [str(meson),"compile","-C",str(build),"-j","2"],
 [str(meson),"install","-C",str(build),"--no-rebuild"]]
if (build/"build.ninja").exists():
    commands=commands[1:]
(receipts/"commands.json").write_text(json.dumps(commands,indent=2))
started=time.monotonic()
for index,command in enumerate(commands):
    print("Executing",command,flush=True)
    with (receipts/f"build-{index}-{int(time.time())}.log").open("w") as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                available=next(int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
                if available<12*1024**2 or time.monotonic()-started>1800:
                    child.terminate()
                    raise RuntimeError("Graphics build resource/deadline guard")
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(f"Graphics build step {index} failed; inspect {receipts}")
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
print("Private driver prefix:",prefix,flush=True)
