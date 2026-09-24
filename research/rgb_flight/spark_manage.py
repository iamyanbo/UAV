"""Explicitly authorized Spark service offload; retain container and weights."""
import json
from pathlib import Path
import subprocess
import time

root=Path.home()/"uav-rgb-flight"
receipt_dir=root/"receipts"
receipt_dir.mkdir(parents=True,exist_ok=True)
inspection=json.loads(subprocess.check_output(["docker","inspect","vllm-fn-tp1"],text=True))[0]
receipt=dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
             container_id=inspection["Id"],name=inspection["Name"],image=inspection["Config"]["Image"],
             restart_policy=inspection["HostConfig"]["RestartPolicy"],
             was_running=inspection["State"]["Running"],
             restart_command="docker start "+inspection["Id"],
             meminfo_before=Path('/proc/meminfo').read_text(),weights_deleted=False)
path=receipt_dir/("model-offload-"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+".json")
path.write_text(json.dumps(receipt,indent=2))
if receipt["was_running"]:
    subprocess.run(["docker","stop","--time","60",inspection["Id"]],check=True,timeout=90)
receipt["running_after"]=json.loads(subprocess.check_output(["docker","inspect","--format","{{json .State.Running}}",inspection["Id"]],text=True))
receipt["meminfo_after"]=Path('/proc/meminfo').read_text()
path.write_text(json.dumps(receipt,indent=2))
print(json.dumps({"receipt":str(path),"running_after":receipt["running_after"],"restart_command":receipt["restart_command"]}))
