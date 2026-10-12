"""Writes the mel filterbank of the export's nemo128.onnx / nemo80.onnx as raw f32 (257 rows of 128
or 80), for the browser page's JS mel front end (onnxruntime-web cannot run those graphs: no float64
Cast kernel).
usage: uv run --no-project --python 3.12 --with onnx --with numpy scripts/mel-filterbank.py cache/onnx-tdt-v2"""
import glob
import sys
import numpy as np
import onnx
from onnx import numpy_helper

d = sys.argv[1]
m = onnx.load(sorted(glob.glob(f"{d}/nemo*.onnx"))[0])
fb = next(numpy_helper.to_array(i) for i in m.graph.initializer if len(i.dims) == 2 and i.dims[0] == 257)
fb.astype("<f4").tofile(f"{d}/melfb-257x{fb.shape[1]}.f32")
print(fb.shape, float(fb.sum()))
