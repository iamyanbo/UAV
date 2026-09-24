"""Bounded real executable launch check, invoked only by the guarded backend."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--renderer", choices=("vulkan", "opengl"), required=True)
    parser.add_argument("--rpc-python")
    parser.add_argument("--display-mode",choices=("offscreen","xvfb"),default="offscreen")
    parser.add_argument("--graphics-driver",choices=("system","dozen","lavapipe","swiftshader"),default="system")
    parser.add_argument("--debug-start",action="store_true")
    parser.add_argument("--disable-rhi-thread",action="store_true")
    parser.add_argument("--validate-graphics",action="store_true")
    args = parser.parse_args()
    output = args.output
    # Mesa's database cache does frequent locked small-file I/O. Keep it on
    # native WSL storage, not the Windows-mounted experiment artifact drive.
    shader_cache=Path.home()/".cache/uav-flight/mesa"
    shader_cache.mkdir(parents=True,exist_ok=True)
    os.environ["MESA_SHADER_CACHE_DIR"]=str(shader_cache)
    (output/"shader_cache.json").write_text(json.dumps({"path":str(shader_cache),"storage":"native WSL filesystem"},indent=2))
    if args.validate_graphics:
        os.environ["VK_INSTANCE_LAYERS"]="VK_LAYER_KHRONOS_validation"
    if args.graphics_driver in ("dozen","lavapipe","swiftshader"):
        prefix="install" if args.graphics_driver=="dozen" else "install-lavapipe"
        pattern="dzn_icd*.json" if args.graphics_driver=="dozen" else "lvp_icd*.json"
        icds=list((Path.home()/"uav-graphics-build"/prefix/"share/vulkan/icd.d").glob(pattern))
        if args.graphics_driver=="swiftshader":
            icds=list((Path.home()/"uav-graphics-build/swiftshader-153.0.8010.52").glob("vk_swiftshader_icd.json"))
        if len(icds)!=1:
            raise RuntimeError("Selected private Vulkan ICD is not installed")
        os.environ["VK_ICD_FILENAMES"]=str(icds[0])
        os.environ["VK_DRIVER_FILES"]=str(icds[0])
        (output/"graphics_environment.json").write_text(json.dumps({key:os.environ[key] for key in ("VK_ICD_FILENAMES","VK_DRIVER_FILES")},indent=2))
    binary = args.scene / "AirVLN/Binaries/Linux/AirVLN-Linux-Shipping"
    if not binary.is_file():
        raise RuntimeError("Selected scene executable missing")
    # Do not modify the distributed settings or the user's Documents/AirSim.
    settings = json.loads((binary.parent / "settings.json").read_text())
    settings.update(ApiServerPort=41451, LocalHostIp="127.0.0.1", RpcEnabled=True, ClockSpeed=1.0, ViewMode="NoDisplay")
    captures = [{"ImageType":0,"Width":640,"Height":480,"FOV_Degrees":90,"MotionBlurAmount":0}]
    settings["CameraDefaults"] = {"CaptureSettings": captures}
    vehicle = settings["Vehicles"]["drone_1"]
    vehicle["Sensors"] = {"Imu":{"SensorType":2,"Enabled":True}}
    vehicle["RC"]["AllowAPIWhenDisconnected"] = True
    vehicle["EnableCollisionPassthrough"] = False
    vehicle["EnableCollisionPassthrogh"] = False  # Actual upstream setting spelling.
    vehicle["Cameras"]["front_custom"]["CaptureSettings"] = captures
    settings_path = output / "settings.json"
    settings_path.write_text(json.dumps(settings,indent=2))
    for command, name in [(["vulkaninfo","--summary"],"vulkaninfo.txt"),(["glxinfo","-B"],"glxinfo.txt"),(["ldd",str(binary)],"ldd.txt")]:
        with (output / name).open("w") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=20)
    from graphics_probe import probe
    capabilities=probe()
    (output/"graphics_capabilities.json").write_text(json.dumps(capabilities,indent=2))
    if capabilities["status"]!="capabilities_available":
        receipt=dict(status=capabilities["status"],graphics_driver=args.graphics_driver,
                     details=capabilities,flight_validated=False,rgb_validated=False,executable_started=False)
        (output/"launch.json").write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)
        return 2
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1",41451)) == 0:
            raise RuntimeError("RPC port already occupied; refusing to test another process")
    binary.chmod(binary.stat().st_mode | 0o111)
    command = [str(binary),"AirVLN","-"+args.renderer,"-RenderOffscreen" if args.display_mode=="offscreen" else "-windowed","-unattended","-nosound",
               "-stdout","-FullStdOutLogOutput","-ResX=640","-ResY=480", "-settings="+str(settings_path),
               "-abslog="+str(output/"unreal.log")]
    if args.display_mode=="xvfb":
        command=["xvfb-run","-a","-s","-screen 0 640x480x24",*command]
    if args.disable_rhi_thread:
        command += ["-norhithread"]
    if args.debug_start:
        command=["gdb","-batch","-ex","set pagination off","-ex","run","-ex","bt 16","-ex","info registers","-ex","x/12i $pc-24","--args",*command]
    (output/"command.json").write_text(json.dumps(command,indent=2))
    started = time.monotonic()
    status = "rpc_not_ready"
    with (output/"simulator_stdout.log").open("w") as log:
        child = subprocess.Popen(command,cwd=args.scene,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None and time.monotonic()-started < 90:
                with socket.socket() as sock:
                    sock.settimeout(.2)
                    if sock.connect_ex(("127.0.0.1",41451)) == 0:
                        status = "rpc_port_open_unvalidated"
                        break
                time.sleep(.5)
            if child.poll() is not None:
                status = "executable_exited_before_rpc"
            elif Path(f"/proc/{child.pid}/maps").exists():
                (output/"loaded_libraries.txt").write_text("\n".join(sorted({row.split()[-1] for row in Path(f"/proc/{child.pid}/maps").read_text().splitlines() if ".so" in row})))
            receipt = {"status":status,"return_code_before_cleanup":child.poll(),"seconds":time.monotonic()-started,
                       "renderer_requested":args.renderer,"graphics_driver":args.graphics_driver,"flight_validated":False,"rgb_validated":False}
            (output/"launch.json").write_text(json.dumps(receipt,indent=2))
            print(json.dumps(receipt),flush=True)
            if status == "rpc_port_open_unvalidated" and args.rpc_python:
                rpc = subprocess.run([args.rpc_python,str(Path(__file__).with_name("rpc_probe.py")),"--output",str(output)],timeout=120)
                if rpc.returncode:
                    status = "rpc_checks_failed"
                receipt["status"] = status
                receipt["rpc_return_code"] = rpc.returncode
                (output/"launch.json").write_text(json.dumps(receipt,indent=2))
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            logs=args.scene/"AirVLN/Saved/Logs"
            if logs.is_dir():
                for path in logs.glob("*.log"):
                    shutil.copy2(path,output/("saved-"+path.name))
    return 0 if status == "rpc_port_open_unvalidated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
