import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import base64
import io
import tempfile

import cv2
import torch
from PIL import Image

import runpod

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
import o_voxel

MODEL_ID = "microsoft/TRELLIS.2-4B"
ENVMAP_PATH = "assets/hdri/forest.exr"  # ships with the TRELLIS.2 repo

# Loaded once per worker at cold start, reused across warm requests
print("Loading TRELLIS.2 pipeline...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
pipeline.cuda()

envmap = EnvMap(
    torch.tensor(
        cv2.cvtColor(cv2.imread(ENVMAP_PATH, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32,
        device="cuda",
    )
)
print("Pipeline ready.")


def handler(job):
    """
    Expected input:
    {
      "input": {
        "image_base64": "<base64-encoded dish photo>",
        "decimation_target": 200000,   # optional, lower = smaller/faster GLB
        "texture_size": 2048           # optional, lower = smaller texture map
      }
    }

    Returns:
    { "glb_base64": "<base64-encoded .glb file>" }
    or
    { "error": "<message>" }
    """
    job_input = job.get("input", {})
    image_b64 = job_input.get("image_base64")
    if not image_b64:
        return {"error": "Missing 'image_base64' in input."}

    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Could not decode input image: {e}"}

    decimation_target = job_input.get("decimation_target", 200000)
    texture_size = job_input.get("texture_size", 2048)

    try:
        mesh = pipeline.run(image)[0]
        mesh.simplify(16777216)  # nvdiffrast vertex limit

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=True) as tmp:
            glb.export(tmp.name, extension_webp=True)
            tmp.seek(0)
            glb_bytes = tmp.read()

        return {"glb_base64": base64.b64encode(glb_bytes).decode("utf-8")}

    except Exception as e:
        return {"error": f"Generation failed: {e}"}


runpod.serverless.start({"handler": handler})
