# TRELLIS.2 RunPod Serverless Worker (internal testing only)

Two files, both go in the same folder:
- `Dockerfile`
- `handler.py`

## ⚠️ Before you build

This image bundles **nvdiffrec**, which is licensed under NVIDIA's
Source Code License-NC — non-commercial research/evaluation use only.
There's an open, unanswered issue on the TRELLIS.2 repo about whether
this makes commercial use of TRELLIS.2 non-compliant. Keep this image
and endpoint **internal-only** (your own testing) until that's resolved
or you've swapped nvdiffrec for something commercially licensed — don't
point live Vexly customer traffic at it yet.

## Build — via GitHub Actions (free, no local machine needed)

This repo includes `.github/workflows/build.yml`, which builds the
image on GitHub's own servers and pushes it straight to Docker Hub.
You don't need Docker installed anywhere yourself.

### One-time setup

1. Create a **new GitHub repository** (public or private) and push
   these files to it (`Dockerfile`, `handler.py`, `README.md`, and the
   `.github/workflows/build.yml` folder).
2. On **Docker Hub** → Account Settings → Security → **New Access
   Token** — create one with Read/Write permissions, and copy it
   (you won't see it again).
3. On your **GitHub repo** → Settings → Secrets and variables →
   Actions → **New repository secret**. Add two secrets:
   - `DOCKERHUB_USERNAME` — your Docker Hub username
   - `DOCKERHUB_TOKEN` — the access token from step 2

### Running the build

Once the files are pushed and the secrets are set, the workflow runs
automatically. You can also trigger it manually: go to your repo's
**Actions** tab → select "Build and push TRELLIS.2 image" → **Run
workflow**.

Expect the first run to take 45-90+ minutes (mostly compiling
flash-attn, cumesh, o-voxel, and flexgemm from source, plus
downloading the model weights). You can watch progress live in the
Actions tab. When it finishes, your image will be at
`<your-dockerhub-username>/trellis2-vexly:latest` on Docker Hub.

## Deploy on RunPod

1. Serverless → **Deploy from a Docker image**
2. Image: `<your-dockerhub-username>/trellis2-vexly:latest`
3. GPU: 24GB+ VRAM required — A100 or H100 (per TRELLIS.2's tested config)
4. Active Workers: 0
5. Flex/Max Workers: 2-3 (keep it capped while testing)
6. Container Disk: give it generous headroom (30GB+) since the baked-in
   weights and dependencies are large

## Test a request

Once the endpoint is live, RunPod gives you a URL like
`https://api.runpod.ai/v2/<endpoint-id>/runsync`. Send a request:

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \
  -H "Authorization: Bearer <your-runpod-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_base64": "<base64 of a test dish photo>"
    }
  }'
```

A successful response returns `{"glb_base64": "..."}` — decode that
back to a `.glb` file to view the generated 3D model.

## Notes

- Cold start on the first request to a new worker will be slow (model
  load into GPU memory + CUDA warmup), independent of RunPod's
  container-level FlashBoot speed.
- `decimation_target` and `texture_size` in the request let you trade
  quality for speed/file-size — useful once you're tuning this for
  actual restaurant dish photos.
