import datetime
from ultralytics import YOLO
from pathlib import Path
import shutil,os
from functools import wraps


def train_only(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "train", False):
            raise RuntimeError("Training is not allowed in inference-only mode.")
        return method(self, *args, **kwargs)
    return wrapper

class Model():
    def __init__(self,path=None,name='yolo11'):
        self.path = path
        self.model_name = name
        if path and os.path.exists(path):
            self.model = YOLO(path)
            self.train = False
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


    
    def infer(self,image, conf = 0.3, return_classes=True):
        results = self.model(image, conf)
        imgs_with_boxes = results[0].plot()
        if not return_classes:
            return imgs_with_boxes
        
        boxes = results[0].boxes
        detected = []
        for box in boxes:
            class_id = int(box.cls)
            label = self.model.names[class_id]
            detected.append({'label': label, 'bbox': box.xyxy.tolist()[0]})
        return imgs_with_boxes, detected