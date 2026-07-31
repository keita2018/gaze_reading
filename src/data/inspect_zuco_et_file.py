#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import h5py, numpy as np
from scipy.io import loadmat

def desc_array(name, arr, lines, indent=''):
    lines.append(f'{indent}{name}: shape={arr.shape} dtype={arr.dtype}')
    if arr.size==0 or not np.issubdtype(arr.dtype, np.number): return
    flat=arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr.ravel()
    if flat.size: lines.append(f'{indent}  min={flat.min()} max={flat.max()}')
    if arr.ndim==1: preview=arr[:8]
    else:
        sl=tuple(slice(0,min(5,s)) for s in arr.shape)
        preview=arr[sl]
    lines.append(f'{indent}  preview={np.array2string(preview, threshold=80)}')
    candidates=[]
    if arr.ndim==1: candidates=[('vector',arr)]
    elif arr.ndim==2:
        candidates += [(f'col[{i}]',arr[:,i]) for i in range(min(arr.shape[1],12))]
        candidates += [(f'row[{i}]',arr[i,:]) for i in range(min(arr.shape[0],12))]
    for label,v in candidates:
        v=np.asarray(v,dtype=float); v=v[np.isfinite(v)]
        if v.size<20: continue
        d=np.diff(v); pos=d[d>0]
        if d.size and np.mean(d>=0)>.95 and pos.size:
            med=float(np.median(pos))
            if .1<=med<=100:
                lines.append(f'{indent}  possible time {label}: median_delta={med:.6g}, approx_hz={1000/med:.3f}')

def is_struct(v): return hasattr(v,'_fieldnames') and v._fieldnames is not None

def walk(v,name,lines,depth=0):
    ind='  '*depth
    if depth>8: lines.append(f'{ind}{name}: <max depth>'); return
    if is_struct(v):
        lines.append(f'{ind}{name}: struct fields={list(v._fieldnames)}')
        for f in v._fieldnames: walk(getattr(v,f),f'{name}.{f}',lines,depth+1)
    elif isinstance(v,np.ndarray):
        if v.dtype==object:
            lines.append(f'{ind}{name}: object array shape={v.shape}')
            for i,x in enumerate(v.ravel()[:20]): walk(x,f'{name}[{i}]',lines,depth+1)
        else: desc_array(name,v,lines,ind)
    elif isinstance(v,(str,bytes,np.str_)):
        s=v.decode(errors='replace') if isinstance(v,bytes) else str(v)
        lines.append(f'{ind}{name}: text={s[:300]!r}')
    else: lines.append(f'{ind}{name}: {type(v).__name__}={repr(v)[:200]}')

def inspect(path):
    lines=[f'FILE: {path.resolve()}']
    if h5py.is_hdf5(path):
        lines.append('FORMAT: HDF5/MATLAB v7.3')
        with h5py.File(path,'r') as h:
            def vis(name,obj):
                if isinstance(obj,h5py.Group): lines.append(f'GROUP /{name} keys={list(obj.keys())[:40]}')
                else:
                    lines.append(f'DATASET /{name} shape={obj.shape} dtype={obj.dtype}')
                    if obj.size<=2000000 and np.issubdtype(obj.dtype,np.number):
                        try: desc_array('/'+name,np.asarray(obj[()]),lines,'  ')
                        except Exception as e: lines.append(f'  read error: {e}')
            h.visititems(vis)
    else:
        lines.append('FORMAT: classic MATLAB MAT')
        d=loadmat(path,squeeze_me=True,struct_as_record=False,chars_as_strings=True)
        keys=[k for k in d if not k.startswith('__')]
        lines.append(f'TOP LEVEL: {keys}')
        for k in keys: walk(d[k],k,lines)
    return lines

def main():
    p=argparse.ArgumentParser(); p.add_argument('mat_files',nargs='+'); p.add_argument('--output-dir',default='logs')
    a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for f in a.mat_files:
        path=Path(f)
        if not path.is_file(): raise FileNotFoundError(path)
        lines=inspect(path)
        rp=out/f'inspect_{path.stem}.txt'; rp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('\n'.join(lines[:100]))
        if len(lines)>100: print(f'... {len(lines)-100} more lines')
        print(f'Written: {rp}')
    return 0
if __name__=='__main__': raise SystemExit(main())
