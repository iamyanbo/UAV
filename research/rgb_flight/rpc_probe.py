"""Privileged engineering diagnostic, never imported by a runtime policy.

No navigation evidence is claimed. Check native RGB, simulation time, physical
velocity commands and collision-query availability before reference collection.
"""
import argparse
import json
import math
from pathlib import Path
import time

import airsim
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-settings",type=Path)
    args=parser.parse_args()
    output=args.output/"engineering_only"
    output.mkdir()
    client=airsim.MultirotorClient(timeout_value=60)
    vehicle="drone_1"
    samples=[]
    receipt={"status":"running","scope":"privileged engineering RPC diagnostic; not a flight episode",
             "pose_setting_calls":0,"pause_calls":0,"collision_detection_validated":False}
    try:
        receipt["operation"]="get_server_version"
        receipt["server_version"]=client.getServerVersion()
        receipt["operation"]="list_vehicles"
        receipt["vehicles"]=client.listVehicles()
        if args.expected_settings:
            receipt["operation"]="verify_effective_settings"
            expected=json.loads(args.expected_settings.read_text())
            effective=json.loads(client.getSettingsString())
            (output/"effective_settings.json").write_text(json.dumps(effective,indent=2))
            for key in ("SimMode","ClockSpeed","ApiServerPort","LocalHostIp","CameraDefaults","Vehicles"):
                if effective.get(key)!=expected.get(key):
                    raise RuntimeError("Effective simulator settings mismatch: "+key)
            receipt["effective_settings_verified"]=True
        receipt["operation"]="check_unpaused_sim_clock"
        if client.simIsPause():
            raise RuntimeError("Physics is paused")
        before=client.getMultirotorState(vehicle_name=vehicle)
        time.sleep(1)
        after=client.getMultirotorState(vehicle_name=vehicle)
        receipt["unpaused_sim_seconds_per_wall_second"]=float(after.timestamp-before.timestamp)/1e9
        # No crop/resizing: the configured camera must actually return 640x480.
        for index in range(102):
            receipt["operation"] = "capture_rgb_" + str(index)
            started=time.monotonic()
            response=client.simGetImages([airsim.ImageRequest("front_custom",airsim.ImageType.Scene,False,False)],vehicle_name=vehicle)[0]
            elapsed=time.monotonic()-started
            sample={"index":index,"sim_timestamp_ns":response.time_stamp,"wall_monotonic_seconds":time.monotonic(),
                    "capture_wall_seconds":elapsed,"width":response.width,"height":response.height,
                    "bytes":len(response.image_data_uint8)}
            samples.append(sample)
            (output/"rgb_samples.json").write_text(json.dumps(samples,indent=2))
            if (response.width,response.height)!=(640,480) or len(response.image_data_uint8)!=640*480*3:
                raise RuntimeError("RGB capture did not return the configured 640x480 RGB byte payload")
            if index in (0,101):
                pixels=np.frombuffer(response.image_data_uint8,dtype=np.uint8).reshape(480,640,3)
                Image.fromarray(pixels).save(output/f"rgb_{index:02d}.png")
        timed=samples[2:]
        receipt["rgb_mean_capture_wall_seconds"]=float(np.mean([x["capture_wall_seconds"] for x in timed]))
        receipt["rgb_wall_hz"]=1/receipt["rgb_mean_capture_wall_seconds"]
        intervals=np.diff([x["sim_timestamp_ns"] for x in timed])/1e9
        receipt["rgb_sim_interval_median_seconds"]=float(np.median(intervals))
        receipt["rgb_sim_interval_p95_seconds"]=float(np.quantile(intervals,.95))
        receipt["rgb_20hz_sim_interval_check"]=bool(np.all(intervals>0) and np.quantile(intervals,.95)<=.05)
        receipt["collision_query_available"]=isinstance(client.simGetCollisionInfo(vehicle_name=vehicle).has_collided,bool)
        # A failed sensor cadence gate does not justify proceeding to flight.
        if not receipt["rgb_20hz_sim_interval_check"]:
            receipt["status"]="blocked_rgb_cadence"
            return 2
        client.enableApiControl(True,vehicle)
        client.armDisarm(True,vehicle)
        client.takeoffAsync(timeout_sec=10,vehicle_name=vehicle).get()
        start=client.getMultirotorState(vehicle_name=vehicle)
        future=client.moveByVelocityBodyFrameAsync(.5,0,0,2,drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                yaw_mode=airsim.YawMode(True,0),vehicle_name=vehicle)
        collisions=[]
        started=time.monotonic()
        while True:
            info=client.simGetCollisionInfo(vehicle_name=vehicle)
            state=client.getMultirotorState(vehicle_name=vehicle)
            collisions.append({"wall_seconds":time.monotonic(),"sim_timestamp_ns":state.timestamp,"collided":info.has_collided,"timestamp":info.time_stamp})
            if (state.timestamp-start.timestamp)/1e9>=2.2:
                break
            if time.monotonic()-started>60:
                raise RuntimeError("Physics failed to advance during motion check")
            time.sleep(.05)
        future.get()
        end=client.getMultirotorState(vehicle_name=vehicle)
        client.hoverAsync(vehicle_name=vehicle).get()
        delta=end.kinematics_estimated.position-start.kinematics_estimated.position
        receipt["physical_displacement_m"]=delta.get_length()
        receipt["motion_sim_seconds"]=float(end.timestamp-start.timestamp)/1e9
        (output/"collision_samples.json").write_text(json.dumps(collisions,indent=2))
        receipt["status"]="blocked_collision_during_motion" if any(x["collided"] for x in collisions) else ("primitive_checks_only" if delta.get_length()>.1 else "blocked_no_physical_motion")
        return 0 if receipt["status"]=="primitive_checks_only" else 2
    except Exception as error:
        receipt.update(status="failed",error_type=type(error).__name__,error=str(error))
        return 2
    finally:
        (output/"rpc.json").write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    raise SystemExit(main())
