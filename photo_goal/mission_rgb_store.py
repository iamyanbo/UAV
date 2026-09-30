"""Canonical RGB provider; v1 file hashes and v2 pixel hashes remain distinct."""
import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
import uuid
from collections import OrderedDict
import numpy as np
from PIL import Image
from .common import digest, contained, FlightLock
from .mission_space import reserve_write


def pixel_hash(pixels):
    if pixels.dtype != np.uint8 or pixels.shape != (480,640,3):
        raise ValueError('Expected calibrated 640x480 uint8 RGB')
    return hashlib.sha256(b'RGB8:640:480\0'+pixels.tobytes()).hexdigest()


class RGBProvider:
    def __init__(self,catalog):
        self.path=Path(catalog).resolve()
        project=os.environ.get('UAV_PROJECT_ROOT')
        if project and not self.path.is_relative_to(Path(project).resolve()):raise ValueError('RGB catalog escapes project root')
        self.root=self.path.parent
        self.cache=OrderedDict()
        self.db=sqlite3.connect(self.path,timeout=10,check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS frames (id TEXT PRIMARY KEY, png TEXT, png_sha TEXT, chunk TEXT, ordinal INTEGER, chunk_sha TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS aliases (path TEXT PRIMARY KEY, id TEXT NOT NULL, deletable INTEGER NOT NULL)')
        self.db.commit()

    def register(self,path,deletable=False):
        path=Path(path).resolve()
        with Image.open(path) as image:
            if image.mode!='RGB' or image.size!=(640,480): raise ValueError('Invalid RGB source')
            pixels=np.asarray(image).copy()
        ident=pixel_hash(pixels)
        relative='frames/'+ident+'.png'
        dest=self.root/relative
        if not self.db.execute('SELECT 1 FROM frames WHERE id=?',(ident,)).fetchone():
            dest.parent.mkdir(exist_ok=True)
            reserve_write(dest,path.stat().st_size+4096)
            if not dest.exists():
                try: os.link(path,dest)
                except OSError:
                    import shutil
                    shutil.copyfile(path,dest)
            self.db.execute('INSERT INTO frames VALUES(?,?,?,NULL,NULL,NULL)',(ident,relative,digest(path)))
        self.db.execute('INSERT OR IGNORE INTO aliases VALUES(?,?,?)',(str(path),ident,int(deletable)))
        self.db.commit()
        return dict(schema='photo-goal-rgb-ref/v2',catalog=str(self.path),pixel_sha256=ident,
                    original_png_sha256=digest(path))

    def resolve(self,ident):
        row=self.db.execute('SELECT png,png_sha,chunk,ordinal,chunk_sha FROM frames WHERE id=?',(ident,)).fetchone()
        if row is None: raise FileNotFoundError('Unknown RGB identity '+ident)
        png,png_sha,chunk,ordinal,chunk_sha=row
        if chunk:
            if chunk not in self.cache:
                path=contained(self.root,chunk)
                if digest(path)!=chunk_sha: raise ValueError('Archived chunk hash mismatch')
                executable=os.environ.get('UAV_FFMPEG')
                if not executable: raise RuntimeError('Archived RGB requires explicit UAV_FFMPEG')
                raw=subprocess.run([executable,'-v','error','-threads','2','-i',str(path),
                    '-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True).stdout
                self.cache[chunk]=np.frombuffer(raw,np.uint8).reshape(-1,480,640,3)
                while len(self.cache)>2:self.cache.popitem(last=False)
            self.cache.move_to_end(chunk)
            pixels=self.cache[chunk][ordinal].copy()
        else:
            path=contained(self.root,png)
            if digest(path)!=png_sha: raise ValueError('Canonical PNG file hash mismatch')
            with Image.open(path) as image:pixels=np.asarray(image).copy()
        if pixel_hash(pixels)!=ident: raise ValueError('Decoded RGB pixel hash mismatch')
        return pixels

    def alias(self,path):
        row=self.db.execute('SELECT id FROM aliases WHERE path=?',(str(Path(path).resolve()),)).fetchone()
        if row is None:raise FileNotFoundError(path)
        return self.resolve(row[0])

    def close(self):self.db.close()

    def flush(self):
        self.db.commit();self.db.execute('PRAGMA wal_checkpoint(FULL)')
        with self.path.open('rb') as stream:os.fsync(stream.fileno())
        if os.name!='nt':
            descriptor=os.open(self.path.parent,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)


def open_rgb(path):
    """Return an owned PIL image, including registered archived source aliases."""
    path=Path(path)
    if path.is_file():
        with Image.open(path) as image:return image.convert('RGB').copy()
    catalog=os.environ.get('UAV_RGB_CATALOG')
    if not catalog:raise FileNotFoundError(path)
    provider=RGBProvider(catalog)
    try:return Image.fromarray(provider.alias(path))
    finally:provider.close()


def archive(catalog,ffmpeg,protected=(),lock=True):
    """Archive only v2 frames; active/pending source aliases must be protected."""
    from contextlib import nullcontext
    from .mission_resources import RunWindow
    window=RunWindow(8)
    provider=RGBProvider(catalog)
    project=Path(os.environ['UAV_PROJECT_ROOT']).resolve()
    protected={str(Path(p).resolve()) for p in protected}
    os.environ['UAV_FFMPEG']=str(Path(ffmpeg).resolve(strict=True))
    if lock and (project/'active-job.lock').exists():
        provider.close();raise RuntimeError('Archive requires an idle campaign or the trainer batch boundary')
    try:
        with FlightLock(project,'city-rgb-archive') if lock else nullcontext():
            # Recover deletion interrupted after a durable catalog commit.
            committed=provider.db.execute('SELECT id,png FROM frames WHERE chunk IS NOT NULL').fetchall()
            for ident,png in committed:
                if not window.admits(30):return
                aliases=provider.db.execute('SELECT path,deletable FROM aliases WHERE id=?',(ident,)).fetchall()
                if not aliases or any(not removable or path in protected for path,removable in aliases):continue
                existing=[Path(path) for path,_ in aliases if Path(path).exists()]
                if existing or contained(provider.root,png).exists():
                    provider.resolve(ident)
                    for path in existing:
                        if not path.resolve().is_relative_to(project):raise ValueError('RGB cleanup escapes project')
                        path.unlink()
                    contained(provider.root,png).unlink(missing_ok=True)
            rows=provider.db.execute('SELECT id,png FROM frames WHERE chunk IS NULL ORDER BY id').fetchall()
            eligible=[]
            for ident,png in rows:
                aliases=provider.db.execute('SELECT path,deletable FROM aliases WHERE id=?',(ident,)).fetchall()
                if not aliases or any(not removable or path in protected for path,removable in aliases):continue
                eligible.append((ident,png,aliases))
            for begin in range(0,len(eligible),128):
                if not window.admits(30):return
                group=eligible[begin:begin+128]
                name='chunks/'+uuid.uuid4().hex+'.mkv'
                output=provider.root/name;output.parent.mkdir(exist_ok=True)
                pending=output.with_suffix('.pending.mkv')
                pixels=[provider.resolve(ident) for ident,_,_ in group]
                raw=b''.join(p.tobytes() for p in pixels)
                reserve_write(pending,len(raw)*2+8*2**20)
                subprocess.run([ffmpeg,'-v','error','-y','-threads','2','-f','rawvideo',
                    '-pixel_format','rgb24','-video_size','640x480','-framerate','20','-i','-',
                    '-c:v','ffv1','-level','3','-pix_fmt','gbrp','-threads','2',str(pending)],
                    input=raw,check=True,capture_output=True)
                decoded=subprocess.run([ffmpeg,'-v','error','-threads','2','-i',str(pending),
                    '-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True).stdout
                if decoded!=raw:raise RuntimeError('FFV1 exact RGB verification failed; originals retained')
                with pending.open('rb') as stream:os.fsync(stream.fileno())
                pending.replace(output);sha=digest(output)
                if os.name!='nt':
                    descriptor=os.open(output.parent,os.O_RDONLY)
                    try:os.fsync(descriptor)
                    finally:os.close(descriptor)
                with provider.db:
                    for ordinal,(ident,_,_) in enumerate(group):
                        provider.db.execute('UPDATE frames SET chunk=?,ordinal=?,chunk_sha=? WHERE id=?',(name,ordinal,sha,ident))
                provider.flush()
                for ident,png,aliases in group:
                    # All aliases here were explicitly registered as disposable v2 flight sources.
                    for source,_ in aliases:
                        path=Path(source).resolve()
                        if not path.is_relative_to(project):raise ValueError('RGB cleanup escapes project')
                        path.unlink(missing_ok=True)
                    contained(provider.root,png).unlink(missing_ok=True)
    finally:provider.close()
