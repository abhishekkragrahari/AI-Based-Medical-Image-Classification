"""CLI inference on one image (research use only).

Usage: python -m scripts.predict --ckpt runs/<run>/best.pt --image xray.png [--explain gradcam --out heat.png]
"""

import argparse
import json
import sys

from medxai.infer import InputError, Predictor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--explain", choices=["gradcam", "ig", "occlusion"])
    ap.add_argument("--out", help="where to save the explanation overlay PNG")
    a = ap.parse_args(argv)
    try:
        pred = Predictor(a.ckpt).predict(a.image, explain_method=a.explain)
    except (InputError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    heat, img = pred.pop("_heatmap", None), pred.pop("_input01", None)
    if heat is not None and a.out:
        import matplotlib.pyplot as plt

        from medxai.xai import overlay

        plt.imsave(a.out, overlay(img, heat))
        pred["explanation_png"] = a.out
    print(json.dumps(pred, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
