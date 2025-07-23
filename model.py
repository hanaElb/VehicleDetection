import datetime
from tkinter import Image
from ultralytics import YOLO
from pathlib import Path
import shutil,os
from functools import wraps
from utils import preprocess
import torch, torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

def train_only(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "train", False):
            raise RuntimeError("Training is not allowed in inference-only mode.")
        return method(self, *args, **kwargs)
    return wrapper

class Model():
    def __init__(self,path=None,name='yolo11', use_quantized=False, class_names=None):
        self.path = path
        self.model_name = name
        self.use_quantized = use_quantized
        self.class_names = class_names  
        self.train = False  # Default to inference mode
        if self.use_quantized:
            if not path or not os.path.exists(path):
                raise ValueError("Quantized model path is invalid or does not exist.")
            self.model = torch.jit.load(self.path)
            self.model.eval()
            
        elif path and os.path.exists(path):
            self.model = YOLO(path)
            self.model.eval()            
        else: # training
            self.model = YOLO(f'{name}.pt')
            self.train = True
        
    @train_only   
    def train(self,
        data_yaml_path,
        epochs=100,
        patience=10,
        img_size=640,
        project="runs/train",
        name=None,
        save_dir="models",
        export=False,
        export_format="onnx",
        quantize=False,
    ):
        
        """
        Train a YOLOv8 model and save the entire experiment folder + export if needed.

        Args:
            model_name (str): e.g., 'yolov8n.pt', 'yolov8s.pt'
            data_yaml_path (str or Path): path to data.yaml
            epochs (int): number of training epochs
            patience (int): early stopping patience
            img_size (int): image resolution
            project (str): where to store YOLO training runs
            name (str): experiment name
            save_dir (str): permanent storage path (e.g., Google Drive)
            export (bool): whether to export the model
            export_format (str): format to export to ('onnx', 'torchscript', etc.)
            quantize (bool): use quantization (only some formats support)
        """
        model = YOLO(self.model_name)
        experiment_name = name or self.model_name.replace(".pt", "") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        # === Train ===
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=img_size,
            patience=patience,
            project=project,
            name=experiment_name,
            verbose=True
        )

        # === Paths ===
        run_path = Path(project) / experiment_name
        save_path = Path(save_dir) / experiment_name
        save_path.mkdir(parents=True, exist_ok=True)

        # === Copy entire experiment folder (weights, results, etc.) ===
        shutil.copytree(run_path, save_path, dirs_exist_ok=True)
        print(f" Saved entire experiment folder to: {save_path}")

        # === Export Model ===
        if export:
            print(f" Exporting model to {export_format} {'(quantized)' if quantize else ''}...")
            model.export(format=export_format, int8=quantize)
            print(f" Exported model to {export_format}")

        return model, results



    def postprocess_quantized(self, out, img_shape, conf_thresh=0.25, iou_thresh=0.45):
        """
        Postprocess YOLO output for quantized models.
        Expects output as [N, 6]: [x1, y1, x2, y2, conf, cls]
        """
        detections = []

        # Convert to CPU tensor and detach if needed
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu()

        # Apply NMS
        if out.ndim == 2 and out.size(1) >= 6:
            # out: [num_detections, 6] => [x1, y1, x2, y2, conf, cls]
            boxes = out[:, :4]
            scores = out[:, 4]
            classes = out[:, 5].int()

            keep_indices = torchvision.ops.nms(boxes, scores, iou_thresh)

            for idx in keep_indices:
                if scores[idx] < conf_thresh:
                    continue

                x1, y1, x2, y2 = boxes[idx].tolist()
                conf = scores[idx].item()
                cls = classes[idx].item()

                detections.append({
                    'label': self.class_names[cls],
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': round(conf, 2)
                })

        return detections

    def infer(self,image, conf = 0.25, return_classes=True):
        if self.use_quantized:
            preprocess_torch = transforms.ToTensor()
            img_tensor = preprocess_torch(image).unsqueeze(0)

            with torch.no_grad():
                raw_out = self.model(img_tensor)[0]  # Might vary by model

            detections = self.postprocess_quantized(raw_out, image.size)

            return image, detections
        
        # Normal inference for non-quantized models
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = preprocess(image)
        results = self.model(image, conf=conf, stream=False, verbose=False)
        img_with_boxes = results[0].plot()
        
        if isinstance(img_with_boxes, np.ndarray):
            img_with_boxes = Image.fromarray(img_with_boxes[:, :, ::-1])

        if not return_classes:
            return img_with_boxes

        boxes = results[0].boxes
        detected = []
        for box in boxes:
            class_id = int(box.cls)
            label = self.model.names[class_id]
            detected.append({'label': label, 'bbox': box.xyxy.tolist()[0]})
        return img_with_boxes, detected