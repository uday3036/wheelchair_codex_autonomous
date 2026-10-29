"""Archive named development outputs, then remove only regenerable caches.

Default is a dry run. --apply verifies every archived byte before removing originals.
Validated comparison directories, canonical source, assets and the venv are retained.
Run only after experiments and notebook kernels have finished writing outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEVELOPMENT=("results/stowage_development","results/autonomous_navigation_checks",
             "results/robustness_development","results/robustness_candidate",
             "results/side_mount_development","results/side_mount_variations","MUJOCO_LOG.TXT")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--archive',type=Path,default=ROOT/'references/development_evidence_20261021.zip')
    args=p.parse_args()
    candidates=[ROOT/name for name in DEVELOPMENT if (ROOT/name).exists()]
    files=sorted({f for path in candidates for f in (path.rglob('*') if path.is_dir() else [path]) if f.is_file() and not f.is_symlink()})
    caches=[]
    for parent in ('src','scripts','tests','notebooks','models','assets','docs','hardware'):
        folder=ROOT/parent
        caches.extend(folder.rglob('__pycache__'))
        caches.extend(folder.rglob('.DS_Store'))
    caches.extend(p for p in (ROOT/'.DS_Store',ROOT/'__pycache__') if p.exists())
    print(f"Development evidence: {len(files)} files, {sum(p.stat().st_size for p in files)/1e6:.2f} MB")
    print('Archive roots:',', '.join(str(p.relative_to(ROOT)) for p in candidates) or 'none')
    print(f'Regenerable cache entries: {len(caches)}')
    if not args.apply:
        print('Dry run only. Use --apply after experiments finish.');return 0
    if files:
        if args.archive.exists():raise FileExistsError('Use a new --archive path; existing archives are never overwritten')
        args.archive.parent.mkdir(parents=True,exist_ok=True)
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        with zipfile.ZipFile(args.archive,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:archive.write(path,str(path.relative_to(ROOT)))
            archive.writestr('archive_checksums.json',json.dumps(hashes,indent=2)+'\n')
        with zipfile.ZipFile(args.archive) as archive:
            for name,expected in hashes.items():
                if hashlib.sha256(archive.read(name)).hexdigest()!=expected:
                    raise RuntimeError(f'Archive verification failed: {name}; originals retained')
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
                    raise RuntimeError(f'File changed during archiving: {name}; originals retained')
        for path in candidates:
            if path.is_dir():shutil.rmtree(path)
            else:path.unlink()
        print('Verified archive:',args.archive)
    for path in caches:
        if not path.exists():continue
        if path.is_dir():shutil.rmtree(path)
        else:path.unlink()
    print('Active source, validated evidence, meshes and environment retained.')
    return 0


if __name__=='__main__':raise SystemExit(main())
