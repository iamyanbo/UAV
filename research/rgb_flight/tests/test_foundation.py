import dataclasses
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acquire
import contracts as c
import preflight as p
import run


class FoundationTests(unittest.TestCase):
    def test_observation_rejects_privileged_fields_and_mutable_payload(self):
        args = dict(episode_id="e", frame_id=0, rgb=bytes(640*480*3), calibration=c.Calibration(320,320,320,240),
                    sim_seconds=1, received_monotonic_seconds=2, command_history=())
        with self.assertRaises(TypeError):
            c.Observation(**args, true_pose=(0,0,0))
        with self.assertRaises(ValueError):
            c.Observation(**{**args, "rgb": bytearray(args["rgb"])})
        obs = c.Observation(**args)
        self.assertFalse(hasattr(obs, "__dict__"))
        self.assertNotIn("labels", {f.name for f in dataclasses.fields(c.Transition)})

    def test_future_commands_and_memory_are_rejected(self):
        with self.assertRaises(ValueError):
            c.Observation("e", 0, bytes(640*480*3), c.Calibration(320,320,320,240), 1, 2,
                          (c.CommandSample(2,c.Command(0,0,0,0)),))
        with self.assertRaises(ValueError):
            c.BeliefSnapshot("e",1,(),(),1,0,2,0)

    def test_diagonal_speed_and_nan_cannot_bypass_limits(self):
        for command in ((3,3,0,0),(float("nan"),0,0,0),(0,0,2,0),(0,0,0,46)):
            with self.assertRaises(ValueError):
                c.Command(*command)

    def test_config_requires_real_episode_scoped_observed_id(self):
        cfg = c.TaskConfig("e","frontier_1","observed_frontier","search",1,1,1,1,"acquire_new_evidence",.7,3)
        cfg.validate_grounding("e", {"frontier_1"}, 2)
        for episode, ids, now in (("e",set(),2),("other",{"frontier_1"},2),("e",{"frontier_1"},4)):
            with self.assertRaises(ValueError):
                cfg.validate_grounding(episode,ids,now)
        with self.assertRaises(ValueError):
            dataclasses.replace(cfg,additional_caution=0)

    def test_redirect_drops_credentials(self):
        req = urllib.request.Request("https://huggingface.co/file",headers={"Authorization":"Bearer private"})
        out = p.SafeRedirect().redirect_request(req,None,302,"",{},"https://cdn.example/file")
        self.assertIsNone(out.get_header("Authorization"))
        with self.assertRaises(ValueError):
            p.SafeRedirect().redirect_request(req,None,302,"",{},"http://cdn.example/file")

    def test_http_failure_receipt_does_not_include_secret_error_body(self):
        class Opener:
            def open(self,*args,**kwargs):
                raise urllib.error.HTTPError("https://example",401,"secret",{"X-Error-Code":"GatedRepo"},io.BytesIO(b"private-token"))
        receipt, raw = p.request("https://example", "private-token", opener=Opener())
        self.assertEqual(receipt,{"http_status":401,"error_code":"GatedRepo"})
        self.assertEqual(raw,b"")

    def test_gated_scene_stops_without_claiming_download_or_flight(self):
        def request(url,token=None,method="GET"):
            if method == "HEAD":
                return {"http_status":401,"error_code":"GatedRepo"}, b""
            if "/tree/" in url:
                return {"http_status":200}, json.dumps([{"path":p.SCENE_FILE,"size":10,"lfs":{"oid":"*"*64}}]).encode()
            return {"http_status":200}, json.dumps({"sha":"a"*40}).encode()
        with patch.object(p,"credentials",return_value=[("cache","private")]):
            receipt = p.probe(Path("."),request_fn=request)
        self.assertEqual(receipt["status"],"blocked")
        self.assertFalse(receipt["downloaded"])
        self.assertIsNone(receipt["published_sha256"])
        self.assertNotIn("private",json.dumps(receipt))

    def test_authenticated_metadata_unredacts_checksum_but_not_flight_gate(self):
        def request(url,token=None,method="GET"):
            if method == "HEAD":
                return {"http_status":200 if token else 401},b""
            if "/tree/" in url:
                return {"http_status":200},json.dumps([{"path":p.SCENE_FILE,"size":10,"lfs":{"oid":("b" if token else "*")*64}}]).encode()
            return {"http_status":200},json.dumps({"sha":"a"*40}).encode()
        with patch.object(p,"credentials",return_value=[("cache","private")]):
            receipt=p.probe(Path("."),request_fn=request)
        self.assertEqual(receipt["status"],"accessible")
        self.assertEqual(receipt["published_sha256"],"b"*64)
        self.assertFalse(receipt["downloaded"])

    def test_checksum_mismatch_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/"file.zip"
            archive.write_bytes(b"test")
            with self.assertRaises(ValueError):
                p.verify_local_archive(archive,"a"*64)
            self.assertEqual(p.verify_local_archive(archive,hashlib.sha256(b"test").hexdigest())["bytes"],4)

    def test_manifest_declares_every_stage_and_no_legacy_substitution(self):
        spec=json.loads((Path(__file__).resolve().parents[1]/"campaign.json").read_text())
        self.assertEqual(tuple(x["name"] for x in spec["stages"]),run.STAGES)
        for stage in spec["stages"]:
            self.assertTrue(all(stage.get(key) for key in ("inputs","outputs","checkpoint","stop","next_command")))
        self.assertFalse(spec["resources"]["toy_fallback"])
        self.assertEqual(spec["models"]["visual"]["backbone"],"V-JEPA 2 ViT-L")

    def test_zip_path_escape_cannot_write_outside_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            archive=base/"bad.zip"
            with zipfile.ZipFile(archive,"w") as stream:
                stream.writestr("../escape","bad")
            with self.assertRaises(ValueError):
                acquire.extract_verified(archive,base/"scene",lambda:None)
            self.assertFalse((base/"escape").exists())

    def test_extract_receipt_binds_file_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            archive=base/"good.zip"
            with zipfile.ZipFile(archive,"w") as stream:
                stream.writestr("scene/file",b"content")
            records=acquire.extract_verified(archive,base/"out",lambda:None)
            self.assertEqual(records[0]["sha256"],hashlib.sha256(b"content").hexdigest())
            self.assertEqual((base/"out/scene/file").read_bytes(),b"content")

    def test_total_gpu_ceiling_and_ram_reserve_stop_before_work(self):
        class Backend:
            @staticmethod
            def gpu_usage(): return 8192,6554
            @staticmethod
            def available_ram(): return 20*1024**3
        with tempfile.TemporaryDirectory() as folder:
            history=[]
            with self.assertRaisesRegex(RuntimeError,"80%"):
                run.resource_guard(Backend,Path(folder),Path(folder),float("inf"),history)
            Backend.gpu_usage=lambda:(8192,2000)
            Backend.available_ram=lambda:11*1024**3
            with self.assertRaisesRegex(RuntimeError,"RAM"):
                run.resource_guard(Backend,Path(folder),Path(folder),float("inf"),history)

    def test_cancel_marker_prevents_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"STOP").write_text("stop")
            with self.assertRaisesRegex(RuntimeError,"STOP"):
                run.resource_guard(None,root,root,float("inf"),[])


if __name__ == "__main__":
    unittest.main()
