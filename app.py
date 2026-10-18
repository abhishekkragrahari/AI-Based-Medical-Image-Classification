"""Local Gradio demo (research use only). Usage: python app.py --ckpt runs/<run>/best.pt

Uploaded images are processed in memory and not stored. Runs on localhost only.
"""

import argparse
import os

# No telemetry / external calls from the demo (DEPLOYMENT.md: privacy).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import numpy as np
from PIL import Image

from medxai.infer import NOTICE, InputError, Predictor
from medxai.xai import overlay


def build(ckpt: str):
    import gradio as gr

    try:
        predictor = Predictor(ckpt)
        load_error = None
    except (FileNotFoundError, ValueError) as e:
        predictor, load_error = None, str(e)

    def run(img: Image.Image, method: str):
        if predictor is None:
            return {"error": load_error}, None
        if img is None:
            return {"error": "please upload an image"}, None
        try:
            out = predictor.predict(img, explain_method=method)
        except InputError as e:
            return {"error": str(e)}, None
        heat, x01 = out.pop("_heatmap"), out.pop("_input01")
        return out, (overlay(x01, heat) * 255).astype(np.uint8)

    with gr.Blocks(title="PneumoniaMNIST XAI demo") as demo:
        gr.Markdown(f"## Chest X-ray classifier with explanations\n**{NOTICE}**")
        with gr.Row():
            inp = gr.Image(type="pil", label="Grayscale chest X-ray (PNG/JPG)")
            heat = gr.Image(label="Explanation (predicted class)")
        method = gr.Radio(["gradcam", "ig", "occlusion"], value="gradcam", label="XAI method")
        out = gr.JSON(label="Scores and model metadata")
        gr.Button("Predict").click(run, [inp, method], [out, heat])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build(a.ckpt).launch(server_name="127.0.0.1", server_port=a.port, share=False)
