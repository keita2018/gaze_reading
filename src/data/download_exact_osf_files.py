#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
from osfclient.api import OSF

def norm(x): return str(x).replace('\\','/').lstrip('/')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-id', default='2urht')
    p.add_argument('--file', action='append', required=True, help='REMOTE=LOCAL')
    p.add_argument('--overwrite', action='store_true')
    a=p.parse_args()
    req={}
    for spec in a.file:
        if '=' not in spec: raise ValueError('--file must be REMOTE=LOCAL')
        r,l=spec.split('=',1); req[norm(r)]=Path(l)
    st=OSF().project(a.project_id).storage('osfstorage')
    found={}
    print(f'Scanning OSF project {a.project_id} ...')
    for f in st.files:
        rp=norm(f.path)
        if rp in req:
            found[rp]=f
            if len(found)==len(req): break
    missing=sorted(set(req)-set(found))
    if missing:
        print('Missing:', file=sys.stderr)
        for m in missing: print('/'+m, file=sys.stderr)
        return 2
    for rp,dst in req.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not a.overwrite:
            print(f'Already exists: {dst}'); continue
        print(f'Downloading /{rp} -> {dst}')
        with dst.open('wb') as fp: found[rp].write_to(fp)
        print(f'{dst.stat().st_size/1048576:.3f} MB')
    return 0

if __name__=='__main__': raise SystemExit(main())
