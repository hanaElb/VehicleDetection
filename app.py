import gradio as gr
from model import Model
from PIL import Image
import numpy as np
import os
from utils import export_and_quantize_model

def main():
    def detect_objects(image, conf_threshold, quantized):
        if image is None:
            return None, "No image uploaded."
    
        normal_model_path = "model/best.pt"
        quantized_model_path = "model/best_quantized.pt"

        if quantized:
            if not os.path.exists(quantized_model_path):
                print("Quantized model not found. Exporting and quantizing...")
                export_and_quantize_model(normal_model_path)
                                        
            model_path = quantized_model_path
        else:
            model_path = normal_model_path

        model = Model(path=model_path, use_quantized=quantized)

        try:
            if isinstance(image, Image.Image):
                image = np.array(image)

            img_with_boxes, detected = model.infer(image, conf=conf_threshold)
            labels = [f"{obj['label']}: {obj['bbox']}" for obj in detected]
            return img_with_boxes, "\n".join(labels)
        except Exception as e:
            return image, f"Error: {str(e)}"

    with gr.Blocks() as demo:
        gr.Markdown("# YOLOv8 Inference")
        gr.Markdown("Upload an image and see predictions. Toggle quantization if needed.")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
            result_image = gr.Image(label="Detection Output")
            result_text = gr.Textbox(label="Detected Objects")

        with gr.Row():
            conf_slider = gr.Slider(0.0, 1.0, value=0.3, label="Confidence Threshold")
            quant_checkbox = gr.Checkbox(label="Use Quantized Model", value=False)

        submit_btn = gr.Button("Run Detection")

        submit_btn.click(
            detect_objects,
            inputs=[image_input, conf_slider, quant_checkbox],
            outputs=[result_image, result_text]
        )

    demo.launch()

if __name__ == "__main__":
    main()
