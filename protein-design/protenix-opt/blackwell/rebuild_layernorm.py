"""Rebuild the pinned stock extension for Ada and consumer Blackwell.

No model, dtype, inference option, optimizer mode, or kernel math is changed.
The image must retain the CUDA toolkit and original kernel sources.
"""
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess

root = Path('/opt/Protenix/protenix/model/layer_norm')
compiler = root / 'torch_ext_compile.py'
source = compiler.read_text()
anchor = '        ("100", "100"),\n'
if source.count(anchor) != 1 or '("120", "120")' in source:
    raise RuntimeError('Unrecognized stock LayerNorm compiler; review the source before rebuilding.')
architectures = subprocess.check_output(['nvcc', '--list-gpu-arch'], text=True)
if 'compute_120' not in architectures:
    raise RuntimeError('This CUDA compiler cannot build consumer Blackwell SM120.')
source_hash = hashlib.sha256(source.encode()).hexdigest()
compiler.write_text(source.replace(anchor, anchor + '        ("120", "120"),\n'))
# Import must build the revised extension rather than loading the old cubin.
(root / 'fast_layer_norm_cuda_v2.so').unlink()
module = importlib.import_module('protenix.model.layer_norm.layer_norm')
binary = Path(module.fast_layer_norm_cuda_v2.__file__)
listing = subprocess.check_output(['cuobjdump', '--list-elf', str(binary)], text=True)
targets = sorted(set(re.findall(r'\.sm_(\d+)\.cubin', listing)), key=int)
if not {'89', '120'}.issubset(targets):
    raise RuntimeError('Rebuilt LayerNorm is missing an Ada or Blackwell native kernel: ' + listing)
manifest = dict(fix='stock-layernorm-sm120-v1', original_compiler_sha256=source_hash,
                compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
                binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                native_targets=['sm_' + target for target in targets])
Path('/opt/mn-protenix-fixes/build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
