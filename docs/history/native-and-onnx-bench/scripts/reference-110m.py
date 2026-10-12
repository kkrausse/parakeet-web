"""Reference transcripts for the 110m model from onnx-asr, the library its ONNX export was made for
(native onnxruntime on the CPU, the library's own nemo80 front end and TDT decoder). Also copies
that front end (nemo80.onnx) next to the model so ort-bench and scripts/mel-filterbank.py can use it.
usage: uv run --no-project --python 3.12 --with onnx-asr --with onnxruntime --with numpy scripts/reference-110m.py MODEL_DIR CLIP.f32 ..."""
import importlib.resources, json, shutil, sys
import numpy as np
import onnx_asr

d, clips = sys.argv[1], sys.argv[2:]
pre = importlib.resources.files("onnx_asr.preprocessors") / "data" / "nemo80.onnx"
shutil.copyfile(str(pre), f"{d}/nemo80.onnx")
for q in (None, "int8"):
    model = onnx_asr.load_model("nemo-conformer-tdt", d, quantization=q, providers=["CPUExecutionProvider"])
    for c in clips:
        text = model.recognize(np.fromfile(c, dtype="<f4"), sample_rate=16000)
        print(json.dumps({"weights": q or "fp32", "clip": c.split("/")[-1].split(".")[0], "text": text}, ensure_ascii=False), flush=True)
