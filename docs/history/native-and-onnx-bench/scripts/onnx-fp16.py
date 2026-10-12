#!/usr/bin/env python3
"""Convert the istupakov encoder ONNX to float16 weights (float32 inputs/outputs kept).

usage: uv run --with onnx --with onnxconverter-common scripts/onnx-fp16.py SRC_DIR DST_DIR [--inline]
--inline writes one .onnx file with the weights inside (for models under protobuf's 2 GB, e.g. the 110m).
Adapted from dictation-server-native-spike/scripts/onnx-fp16.py (commit dcfa00e).
"""
import os
import sys

import onnx
from onnxconverter_common import float16

inline = "--inline" in sys.argv
source, target = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.makedirs(target, exist_ok=True)
# The model is over protobuf's 2 GB limit, so shape inference has to run on disk; the converter needs
# the value types or it leaves float32/float16 mismatches.
inferred = os.path.join(source, "encoder.inferred.onnx")
onnx.shape_inference.infer_shapes_path(os.path.join(source, "encoder-model.onnx"), inferred)
model = onnx.load(inferred)
os.remove(inferred)
model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
# The converter retypes value_info but leaves the graph's own Cast(to=FLOAT) nodes, which ORT rejects.
half = {info.name for info in model.graph.value_info if info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16}
recast = 0
for node in model.graph.node:
    if node.op_type == "Cast" and node.output[0] in half:
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                attribute.i = onnx.TensorProto.FLOAT16
                recast += 1
print(f"recast {recast} Cast nodes to float16")
if inline:
    onnx.save(model, os.path.join(target, "encoder-model.onnx"))
else:
    onnx.save(model, os.path.join(target, "encoder-model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
              location="encoder-model.onnx.data", size_threshold=1024)
for name in os.listdir(source):
    link = os.path.join(target, name)
    if not name.startswith("encoder") and not os.path.lexists(link):
        os.symlink(os.path.join(source, name), link)
print(sorted((name, os.path.getsize(os.path.join(target, name))) for name in os.listdir(target)))
