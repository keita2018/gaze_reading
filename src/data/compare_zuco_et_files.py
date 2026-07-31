#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import h5py, numpy as np
from scipy.io import loadmat

def is_struct(v): return hasattr(v,'_fieldnames') and v._fieldnames is not None
def walk(v,prefix,out,depth=0):
    if depth>8:return
    if is_struct(v):
        out[prefix]=('struct',','.join(v._fieldnames))
        for f in v._fieldnames: walk(getattr(v,f),prefix+'.'+f,out,depth+1)
    elif isinstance(v,np.ndarray):
        out[prefix]=(f'ndarray {v.shape}',str(v.dtype))
        if v.dtype==object:
            for i,x in enumerate(v.ravel()[:10]): walk(x,f'{prefix}[{i}]',out,depth+1)
    else: out[prefix]=(type(v).__name__,repr(v)[:100])

def tree(path):
    out={}
    if h5py.is_hdf5(path):
        with h5py.File(path,'r') as h:
            def vis(name,obj):
                out['/'+name]=('group',','.join(obj.keys())) if isinstance(obj,h5py.Group) else (f'dataset {obj.shape}',str(obj.dtype))
            h.visititems(vis)
    else:
        d=loadmat(path,squeeze_me=True,struct_as_record=False)
        for k,v in d.items():
            if not k.startswith('__'): walk(v,k,out)
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('raw'); p.add_argument('corrected'); p.add_argument('--output',default='logs/compare_ET_raw_corrected.txt')
    a=p.parse_args(); r=tree(Path(a.raw)); c=tree(Path(a.corrected)); lines=[]
    for k in sorted(set(r)|set(c)):
        if r.get(k)!=c.get(k):
            lines += [f'PATH: {k}',f'  RAW: {r.get(k)}',f'  CORRECTED: {c.get(k)}']
    if not lines: lines=['No structural differences found.']
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:120])); print(f'Written: {out}')
    return 0
if __name__=='__main__': raise SystemExit(main())
