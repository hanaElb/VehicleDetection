import gradio as gr
from model import Model  # assuming your code is saved in model.py
from PIL import Image
import numpy as np

def main():
    

    # Initialize model (change path if needed or pass None to load default weights)
    model = Model(path="model/best.pt")  # or path=None if you want to load a default .pt

    def detect_objects(image, conf_threshold):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        try:
            img_with_boxes, detected = model.infer(image, conf=conf_threshold)
            labels = [f"{obj['label']}: {obj['bbox']}" for obj in detected]
            return img_with_boxes, "\n".join(labels)
        except Exception as e:
            return image, f"Error: {str(e)}"

    # Gradio interface
    interface = gr.Interface(
        fn=detect_objects,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Slider(0.0, 1.0, value=0.3, label="Confidence Threshold")
        ],
        outputs=[
            gr.Image(label="Detection Output"),
            gr.Textbox(label="Detected Objects")
        ],
        title="YOLOv8 Inference",
        description="Upload an image and see YOLOv8 predictions"
    )

    interface.launch()


if __name__ == "__main__":
    main()