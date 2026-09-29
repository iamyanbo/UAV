"""Interactive Spark credential setup; never pass tokens through chat or argv."""
import getpass
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def main():
    token=getpass.getpass('Hugging Face read token (hidden): ').strip()
    if not token:raise SystemExit('No token supplied; nothing changed.')
    headers={'Authorization':'Bearer '+token,'User-Agent':'photo-map-research/1'}
    try:
        with urllib.request.urlopen(urllib.request.Request(
                'https://huggingface.co/api/whoami-v2',headers=headers),timeout=30) as response:
            json.load(response)
        url='https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen/resolve/main/airsim/env_airsim_18.zip'
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers,method='HEAD'),timeout=30):pass
    except urllib.error.HTTPError as error:
        raise SystemExit(f'Access check failed (HTTP {error.code}). Accept dataset terms and check token read access; nothing saved.')
    except Exception:
        raise SystemExit('Network verification failed; nothing saved.')
    home=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))
    home.mkdir(parents=True,exist_ok=True)
    path=home/'token'
    if path.exists():
        raise SystemExit('An existing token is present; preserved it. Configure the account using your Hugging Face CLI instead.')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream:stream.write(token)
    print('OpenFly download access verified and read token saved privately on Spark.')


if __name__=='__main__':main()
