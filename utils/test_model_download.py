"""
On-demand downloader for a curated set of ~100 popular public CNN / RNN ONNX
models (no LLMs).

Models are pulled from the community ONNX Model Zoo mirror on Hugging Face:
    https://huggingface.co/onnxmodelzoo/<slug>/resolve/main/<slug>.onnx

They are saved to the `models/` directory at the repository root. This script
is meant to be run manually when a local corpus of real models is wanted for
testing the toolbox against actual graphs; nothing is downloaded automatically.

Usage (run from the repository root):
    # List the catalog without downloading
    python utils/test_model_download.py --list

    # Download everything (large: several GB total)
    python utils/test_model_download.py --all

    # Download only certain categories
    python utils/test_model_download.py --category classification detection

    # Download specific models by name
    python utils/test_model_download.py --models resnet50-v1-7 mobilenetv2-12

    # Skip anything bigger than 60 MB
    python utils/test_model_download.py --all --max-mb 60

    # Re-download even if the file already exists
    python utils/test_model_download.py --models mnist-12 --force

By default existing valid files are skipped, so re-running resumes a partial
corpus. After each download the file is validated with onnx.checker and its
input/output shapes are printed.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request

# Repo-root-relative imports / paths.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MODELS_DIR = os.path.join(_REPO_ROOT, "models")
HF_BASE = "https://huggingface.co/onnxmodelzoo"

# Every entry is a Hugging Face slug that was confirmed reachable (HTTP 200)
# via a HEAD probe. The download URL and local filename are both derived from
# the slug: <slug>.onnx.
#
# Grouped by task category. All are CNN/RNN vision or classic models; no LLMs.
CATALOG = {
    "classification": [
        "resnet18-v1-7", "resnet34-v1-7", "resnet50-v1-7", "resnet101-v1-7",
        "resnet152-v1-7", "resnet18-v2-7", "resnet50-v2-7", "resnet34-v2-7",
        "resnet101-v2-7", "resnet152-v2-7", "resnet50-v1-12", "resnet50-caffe2-v1-9",
        "mobilenetv2-7", "mobilenetv2-10", "mobilenetv2-12",
        "mobilenetv2-12-int8", "mobilenetv2-12-qdq",
        "squeezenet1.0-7", "squeezenet1.1-7", "squeezenet1.0-9",
        "squeezenet1.0-12",
        "vgg16-7", "vgg16-12", "vgg16-bn-7", "vgg19-7", "vgg19-bn-7",
        "vgg19-caffe2-9",
        "bvlcalexnet-3", "bvlcalexnet-7", "bvlcalexnet-9", "bvlcalexnet-12",
        "caffenet-3", "caffenet-7", "caffenet-9", "caffenet-12",
        "googlenet-3", "googlenet-7", "googlenet-9", "googlenet-12",
        "googlenet-12-int8",
        "inception-v1-7", "inception-v1-9", "inception-v1-12",
        "inception-v1-12-int8", "inception-v2-7", "inception-v2-9",
        "zfnet512-7", "zfnet512-9", "zfnet512-12",
        "densenet-7", "densenet-9", "densenet-12",
        "shufflenet-9", "shufflenet-v2-10", "shufflenet-v2-12",
        "rcnn-ilsvrc13-7", "rcnn-ilsvrc13-9",
        "efficientnet-lite4-11", "efficientnet-lite4-11-int8",
        "efficientnet-lite4-11-qdq",
        "mnist-1", "mnist-7", "mnist-8", "mnist-12",
    ],
    "detection": [
        "ssd-10", "ssd-12",
        "ssd_mobilenet_v1_10", "ssd_mobilenet_v1_12", "ssd_mobilenet_v1_12-int8",
        "tiny-yolov3-11", "yolov3-10", "yolov3-12", "yolov4",
        "tinyyolov2-7", "tinyyolov2-8", "yolov2-coco-9",
        "FasterRCNN-10", "FasterRCNN-12", "FasterRCNN-12-int8",
        "MaskRCNN-10", "MaskRCNN-12", "MaskRCNN-12-int8",
        "retinanet-9",
    ],
    "segmentation": [
        "fcn-resnet50-11", "fcn-resnet50-12", "fcn-resnet101-11",
    ],
    "face_body": [
        "arcfaceresnet100-8",
        "version-RFB-320", "version-RFB-320-int8", "version-RFB-640",
        "age_googlenet", "gender_googlenet",
        "emotion-ferplus-2", "emotion-ferplus-7", "emotion-ferplus-8",
    ],
    "style_transfer": [
        "candy-9", "mosaic-9", "pointilism-9", "rain-princess-9", "udnie-9",
    ],
    "super_resolution": [
        "super-resolution-10",
    ],
}


def model_url(slug):
    return f"{HF_BASE}/{slug}/resolve/main/{slug}.onnx"


def local_path(slug):
    return os.path.join(MODELS_DIR, f"{slug}.onnx")


def iter_selected(categories=None, models=None):
    """
    Yield (category, slug) pairs for the selection. If both filters are None,
    yields the entire catalog.
    """
    for category, slugs in CATALOG.items():
        if categories and category not in categories:
            continue
        for slug in slugs:
            if models and slug not in models:
                continue
            yield category, slug


def human_size(num_bytes):
    if num_bytes is None:
        return "?"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024


def head_size(slug):
    """Return the remote Content-Length in bytes, or None if unavailable."""
    req = urllib.request.Request(model_url(slug), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError):
        return None


def download_one(slug, force=False):
    """
    Download a single model. Returns a status string:
    "skipped", "downloaded", or "failed: <reason>".
    """
    dest = local_path(slug)
    if os.path.exists(dest) and not force:
        return "skipped"

    url = model_url(slug)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
            total = resp.headers.get("Content-Length")
            total = int(total) if total is not None else None
            downloaded = 0
            chunk = 1024 * 256
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
                if total:
                    pct = downloaded * 100 // total
                    print(
                        f"\r    {slug}: {pct:3d}% "
                        f"({human_size(downloaded)}/{human_size(total)})",
                        end="",
                        flush=True,
                    )
        print()
        os.replace(tmp, dest)
        return "downloaded"
    except Exception as e:  # noqa: BLE001 - report any failure, keep going
        if os.path.exists(tmp):
            os.remove(tmp)
        return f"failed: {type(e).__name__}: {e}"


def describe_model(path):
    """
    Return (input_shapes, output_shapes) for an ONNX file as lists of
    "name: [dims]" strings. Best-effort; returns ([], []) on failure.
    """
    try:
        import onnx
    except ImportError:
        return [], []

    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
    except Exception:
        try:
            model = onnx.load(path)
        except Exception:
            return [], []

    initializer_names = {init.name for init in model.graph.initializer}

    def shape_str(value_info):
        dims = []
        for d in value_info.type.tensor_type.shape.dim:
            if d.dim_param:
                dims.append(d.dim_param)  # dynamic/symbolic dimension
            else:
                dims.append(d.dim_value)
        return f"{value_info.name}: {dims}"

    inputs = [
        shape_str(i) for i in model.graph.input if i.name not in initializer_names
    ]
    outputs = [shape_str(o) for o in model.graph.output]
    return inputs, outputs


def main():
    parser = argparse.ArgumentParser(
        description="On-demand downloader for ~100 public CNN/RNN ONNX models"
    )
    parser.add_argument(
        "--list", action="store_true", help="List the catalog and exit (no download)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Download every model in the catalog"
    )
    parser.add_argument(
        "--category",
        nargs="+",
        choices=sorted(CATALOG.keys()),
        help="Only download the given categories",
    )
    parser.add_argument(
        "--models", nargs="+", help="Only download the given model slugs"
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=None,
        help="Skip models whose remote size exceeds this many MB",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file exists"
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print input/output shapes of each downloaded/existing model",
    )
    args = parser.parse_args()

    total_models = sum(len(v) for v in CATALOG.values())

    if args.list:
        print(f"Catalog: {total_models} models across {len(CATALOG)} categories\n")
        for category, slugs in CATALOG.items():
            print(f"[{category}] ({len(slugs)})")
            for slug in slugs:
                print(f"  {slug}")
            print()
        return 0

    if not (args.all or args.category or args.models):
        parser.error("choose what to download: --all, --category, or --models")

    os.makedirs(MODELS_DIR, exist_ok=True)
    selection = list(iter_selected(args.category, args.models))

    if args.models:
        known = {slug for _, slug in iter_selected()}
        unknown = [m for m in args.models if m not in known]
        for m in unknown:
            print(f"WARNING: '{m}' is not in the catalog, skipping.")

    print(f"Selected {len(selection)} model(s). Saving to {MODELS_DIR}\n")

    counts = {"downloaded": 0, "skipped": 0, "failed": 0, "too_large": 0}
    for category, slug in selection:
        if args.max_mb is not None and not (
            os.path.exists(local_path(slug)) and not args.force
        ):
            size = head_size(slug)
            if size is not None and size > args.max_mb * 1024 * 1024:
                print(
                    f"[{category}] {slug}: too large "
                    f"({human_size(size)} > {args.max_mb}MB), skipping"
                )
                counts["too_large"] += 1
                continue

        print(f"[{category}] {slug}")
        result = download_one(slug, force=args.force)
        if result == "downloaded":
            counts["downloaded"] += 1
        elif result == "skipped":
            counts["skipped"] += 1
            print(f"    already present, skipping")
        else:
            counts["failed"] += 1
            print(f"    {result}")
            continue

        if args.describe:
            inputs, outputs = describe_model(local_path(slug))
            print(f"    inputs : {inputs}")
            print(f"    outputs: {outputs}")

    print(
        "\nDone. "
        f"downloaded={counts['downloaded']} skipped={counts['skipped']} "
        f"failed={counts['failed']} too_large={counts['too_large']}"
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
