# Cambodia Traffic Signs Detection — Vercel trial

This folder packages the existing Flask traffic-sign app for Vercel's native Python runtime. It includes both trained models, all three example photos, and the browser interface. No Dockerfile is needed.

## Deploy

1. Create a new public GitHub repository and upload the **contents of this folder** to its root. Do not upload the entire CAMTS research repository or unrelated datasets. Check that `app.py`, `pyproject.toml`, `uv.lock`, `.python-version`, `vercel.json`, `templates/`, `public/`, and `models/` appear at the repository root.
2. Import the new repository in the [Vercel dashboard](https://vercel.com/new). If Vercel asks for a framework preset, choose **Other** (it may also detect Flask automatically).
3. The app runs the INT8 ONNX models with ONNX Runtime on CPU; PyTorch, torchvision, and Ultralytics are not dependencies. Redeploy using the committed `pyproject.toml` and `uv.lock`.
4. Deploy with the default build/install settings. `vercel.json` tells Vercel to include the model checkpoints and templates and allows a request up to Hobby's five-minute maximum. If using a monorepo instead of a new repository, set the Vercel project's **Root Directory** to `share-vercel`.
5. Open the deployment link and try a sample image. Check the Vercel **Build Logs**, **Functions logs**, and **Usage** page if the deployment fails or inference times out.

## Important Vercel limits

- **Trial only:** Vercel Hobby is restricted to personal, non-commercial use. If this is for work/colleagues, confirm your organisation is permitted to use Hobby or use an appropriate plan.
- Python Function requests and responses have a hard **4.5 MB** body limit. This trial caps input images at 3 MB and reduces the annotated result so its response can fit. Very large photos must be resized before upload. Browser-sent video/webcam frames are JPEG-compressed and resized before detection.
- A standard Python function bundle is limited to **500 MB uncompressed**. Using ONNX Runtime instead of PyTorch removes the previous multi-gigabyte inference dependency stack; this bundle should fit within the standard limit.
- Hobby functions have **1 vCPU and 2 GB memory**, and at most **4 CPU-hours** and **360 GB-hours provisioned memory** in the usual monthly usage guidelines. Each sampled video/webcam frame triggers another function request. Model cold starts and CPU inference can therefore remain slow, and frequent video use consumes the small CPU allowance quickly.
- The Hobby function request limit is **300 seconds**, not a promise that the request will finish that quickly.
- The deployment and the included model weights are public. Uploaded media is sent to Vercel for processing; do not use private or sensitive material.
- Vercel Hobby is not a free GPU host. Serving page assets from its CDN does not accelerate CPU model inference.

The INT8 detector accepts letterboxed `640×640` RGB inputs and returns raw YOLO boxes; the app applies confidence filtering, non-maximum suppression, and source-image coordinate scaling. The INT8 sign classifier accepts center-cropped `64×64` RGB detection crops. Uploaded images run both models; video/webcam frames lazily load and run only the detector. Class labels are read from ONNX model metadata.

See Vercel's current [Flask deployment guide](https://vercel.com/docs/frameworks/backend/flask), [Python runtime](https://vercel.com/docs/functions/runtimes/python), [Function limits](https://vercel.com/docs/functions/limitations), [Hobby limits](https://vercel.com/docs/plans/hobby), and [fair-use/commercial-usage guidance](https://vercel.com/docs/limits/fair-use-guidelines#commercial-usage). Limits and plan rules can change; check your Vercel dashboard and account terms before sharing the deployment.
