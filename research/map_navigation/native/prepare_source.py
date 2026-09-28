"""Deferred source preparation only; never downloads, configures or builds.

Adds a single-step mapper entry point to the pinned checkout. The upstream
offline run() remains available for historical tools, but is never used live.
"""
import argparse
from pathlib import Path
import subprocess

PIN = 'f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac'


def prepare(root):
    root=Path(root)
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=PIN:
        raise ValueError('Photo-SLAM revision mismatch')
    header=root/'include/gaussian_mapper.h';source=root/'src/gaussian_mapper.cpp'
    h=header.read_text();s=source.read_text()
    if 'liveStep' in h:raise ValueError('Already prepared; inspect existing patch')
    start=s.index('void GaussianMapper::run()')
    end=s.index('void GaussianMapper::trainColmap()',start)
    run=s[start:end]
    # Reuse the exact upstream initialization (camera setup, sparse points,
    # Gaussian optimizer); strip its unbounded loops and offline finalization.
    a=run.index('            pSLAM_->getAtlas()->clearMappingOperation();')
    b=run.index('            // Invoke training once',a)
    init=run[a:b]
    method='''
bool GaussianMapper::liveStep() {
    if (pSLAM_->isShutDown() || isStopped()) return false;
    live_mode_ = true;
    training_report_interval_ = keyframe_record_interval_ = all_keyframes_record_interval_ = 0;
    // Bound growth: sparse supported points can still be incorporated, but
    // unbounded synthetic densification is disabled for the flight adapter.
    opt_params_.densify_until_iter_ = 0;
    if (!initial_mapped_) {
        if (!hasMetInitialMappingConditions()) return false;
INIT
        initial_mapped_ = true;
    } else if (hasMetIncrementalMappingConditions()) {
        combineMappingOperations();
        if (cull_keyframes_) cullKeyframes();
    }
    trainForOneIteration();
    return true;
}
'''.replace('INIT',init)
    write_call='writeKeyframeUsedTimes(result_dir_ / "used_times");'
    if s.count(write_call)!=1:raise ValueError('Upstream training logging changed')
    s=s.replace(write_call,'if (!live_mode_) '+write_call)
    queue='while (pSLAM_->getAtlas()->hasMappingOperation()) {'
    if s.count(queue)!=1:raise ValueError('Upstream mapping queue changed')
    s=s.replace(queue,'int live_operations = 0;\n    while (pSLAM_->getAtlas()->hasMappingOperation() && (!live_mode_ || live_operations++ < 1)) {')
    header.write_text(h.replace('    void run();','    void run();\n    bool liveStep();\n    bool live_mode_ = false;',1))
    source.write_text(s+'\n'+method)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkout');prepare(p.parse_args().checkout)
