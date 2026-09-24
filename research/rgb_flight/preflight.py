"""Read-only scene/access feasibility gate. No simulator or model fallback.

Only metadata and HEAD requests are sent. Credentials and signed redirect URLs
are never included in receipts. A successful access probe is NOT flight evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = "IPEC-COMMUNITY/OpenFly_DataGen"
SCENE_FILE = "airsim/env_airsim_16.zip"
API = "https://huggingface.co/api/datasets/" + REPOSITORY


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # CDN requests do not need the account credential. Never forward it.
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("Refusing non-HTTPS asset redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def credentials(root: Path):
    """Only standard HF credential locations; never enumerate unrelated secrets."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            yield name, os.environ[name].strip()
    paths = []
    if os.environ.get("HF_TOKEN_PATH"):
        paths.append(Path(os.environ["HF_TOKEN_PATH"]))
    if os.environ.get("HF_HOME"):
        paths.append(Path(os.environ["HF_HOME"]) / "token")
    paths += [Path.home() / ".cache/huggingface/token", root / "cache/huggingface/token"]
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                yield "local_hf_cache", value


def request(url, token=None, method="GET", opener=None):
    headers = {"User-Agent": "rgb-flight-preflight/1"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers, method=method)
    opener = opener or urllib.request.build_opener(SafeRedirect())
    try:
        with opener.open(req, timeout=20) as response:
            data = response.read(4 * 1024 * 1024) if method == "GET" else b""
            return {"http_status": response.status}, data
    except urllib.error.HTTPError as error:
        # Error bodies can contain request data; save only a known error code.
        code = error.headers.get("X-Error-Code", "")
        error.close()
        return {"http_status": error.code, "error_code": code if code in
                {"GatedRepo", "RepoNotFound", "EntryNotFound", "RevisionNotFound"} else "HTTPError"}, b""
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"http_status": None, "error_code": "NetworkError"}, b""


def probe(root: Path, guard=lambda: None, request_fn=request):
    result = {"repository": REPOSITORY, "scene_file": SCENE_FILE,
              "status": "blocked", "attempts": [], "downloaded": False}
    # Public metadata is safe to inspect before authenticating. Pin revision.
    guard()
    metadata_status, raw = request_fn(API)
    result["metadata_request"] = metadata_status
    if metadata_status.get("http_status") != 200:
        result["reason"] = "Repository metadata unavailable; cannot pin scene revision"
        return result
    metadata = json.loads(raw)
    revision = metadata.get("sha", "")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        result["reason"] = "Repository returned no valid immutable revision"
        return result
    result["revision"] = revision
    url = API + "/tree/" + revision + "/airsim?limit=100"
    guard()
    listing_status, raw = request_fn(url)
    result["listing_request"] = listing_status
    if listing_status.get("http_status") != 200:
        result["reason"] = "Pinned scene listing unavailable"
        return result
    entries = [row for row in json.loads(raw) if row.get("path") == SCENE_FILE]
    if len(entries) != 1:
        result["reason"] = "Selected archive absent from pinned listing"
        return result
    entry = entries[0]
    result["archive_bytes"] = entry["size"]
    digest = entry.get("lfs", {}).get("oid", "")
    result["published_sha256"] = digest if re.fullmatch(r"[a-f0-9]{64}", digest) else None
    url = "https://huggingface.co/datasets/" + REPOSITORY + "/resolve/" + revision + "/" + SCENE_FILE
    result["source_url"] = url
    used_tokens = set()
    for source, token in [("anonymous", None), *credentials(root)]:
        if token in used_tokens:
            continue
        used_tokens.add(token)
        guard()
        status, _ = request_fn(url, token=token, method="HEAD")
        result["attempts"].append({"credential_source": source, **status})
        if status.get("http_status") == 200:
            # Gated public listings redact LFS hashes. Repeat with the working
            # credential to obtain the published checksum, without persisting it.
            guard()
            detail_status, detail_raw = request_fn(API + "/tree/" + revision + "/airsim?limit=100", token=token)
            if detail_status.get("http_status") == 200:
                detail = next((x for x in json.loads(detail_raw) if x.get("path") == SCENE_FILE), {})
                digest = detail.get("lfs", {}).get("oid", "")
                result["published_sha256"] = digest if re.fullmatch(r"[a-f0-9]{64}", digest) else None
            result["status"] = "accessible"
            result["reason"] = "Archive access verified; download and actual flight validation remain required"
            return result
    result["reason"] = "Scene archive unavailable with available credentials; no substitute environment permitted"
    return result


def verify_local_archive(path: Path, expected_sha256: str, guard=lambda: None):
    """An authorized local copy still needs a trusted expected digest."""
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("A trusted SHA-256 is required for a local scene archive")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            guard()
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError("Local scene archive SHA-256 mismatch")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}
