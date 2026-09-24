"""Build native research dependencies in a dedicated, owned container."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--resume-container')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--dependencies-only',action='store_true')
    args = parser.parse_args()
    if args.inside:
        env = dict(os.environ, TORCH_CUDA_ARCH_LIST='12.1', MAX_JOBS='4', PIP_DISABLE_PIP_VERSION_CHECK='1')
        packages = ['av==18.1.0','gdown==5.2.0','transformers==4.57.1','peft==0.17.1','qwen-vl-utils==0.0.14',
                    'timm==1.0.22','einops==0.8.1','plyfile==1.1.3','munch==4.0.0',
                    'evo==1.34.0','imgviz==1.7.6','lpips==0.1.4','kornia==0.8.2',
                    'open3d==0.20.0','pyrender==0.1.45','colorama==0.4.6','rich==14.2.0','opencv-python-headless==4.11.0.86']
        if not args.verify_only:
            subprocess.run(['python','-m','pip','install',*packages],env=env,check=True)
        modules = ['thirdparty/lietorch','thirdparty/simple-knn','thirdparty/diff-gaussian-rasterization-w-pose','.']
        if not args.verify_only and not args.dependencies_only:
            for module in modules:
                subprocess.run(['python','-m','pip','install','--no-build-isolation','--no-deps','-v',str(Path('/port')/module)],env=env,check=True)
            subprocess.run(['python','-m','pip','install','--no-build-isolation','torch-scatter==2.1.2'],env=env,check=True)
        Path('/output/model-packages.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
        subprocess.run(['python','/source/splat_kernel_check.py'],check=True)
        return
    root = Path.home()/'uav-rgb-flight'
    image = json.loads((root/'receipts/model-image.json').read_text())['image']
    if args.dependencies_only:
        image=json.loads((root/'receipts/model-stack.json').read_text())['image_id']
    if args.resume_container:
        if not args.resume_container.startswith('rgb-model-build-') or not args.resume_container[16:].isdigit():
            raise ValueError('Only an owned model-build container can supply a dependency cache')
        prior=json.loads(subprocess.check_output(['docker','inspect',args.resume_container],text=True))[0]
        if prior['State']['Running']:
            raise RuntimeError('Cannot cache a running build')
        image=subprocess.check_output(['docker','commit',args.resume_container],text=True).strip()
    if not (root/'receipts/splat-native-port.json').is_file():
        raise RuntimeError('Prepare and record the source port before building')
    source = Path(__file__).resolve().parent
    job = Path(os.environ['RGB_JOB_DIR'])
    name = 'rgb-model-build-'+str(os.getpid())
    def terminate(signum, frame):
        raise KeyboardInterrupt('Build cancelled')
    signal.signal(signal.SIGTERM,terminate)
    try:
        subprocess.run(['docker','run','--name',name,'--label','rgb-flight.job='+str(job),'--gpus','all','--shm-size','2g',
                        '-v',str(source)+':/source:ro','-v',str(root/'ports/Splat-SLAM')+':/port',
                        '-v',str(job)+':/output', image,'python','/source/build_model_stack.py','--inside',
                        *(['--verify-only'] if args.verify_only else []),
                        *(['--dependencies-only'] if args.dependencies_only else [])],check=True)
        built = subprocess.check_output(['docker','commit',name,'rgb-flight-models:25.11-native'],text=True).strip()
        receipt=dict(status='native_extensions_checked; selected_models_not_yet_verified',image_id=built,base_image=image,
                     port_receipt=str(root/'receipts/splat-native-port.json'),job=str(job))
        (root/'receipts/model-stack.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)
    finally:
        subprocess.run(['docker','stop','-t','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        # Preserve failed build containers/logs for diagnosis; no automatic prune.


if __name__=='__main__':
    main()
