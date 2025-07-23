from PIL import Image, ImageOps
import torch
from ultralytics import YOLO
import os

def preprocess(image, target_size=640):
    image = image.convert("RGB")
    image = ImageOps.exif_transpose(image)

    if max(image.size) > target_size:
        image = image.resize((target_size, target_size))  # Resize only if too big

    return image


def export_and_quantize_model(pt_path: str,
                              export_dir: str = "model",
                              scripted_name: str = "best_scripted.torchscript",
                              quantized_name: str = "best_quantized.pt"):
    """
    Export a YOLOv8 model to TorchScript and apply dynamic quantization.

    Args:
        pt_path (str): Path to the YOLOv8 `.pt` model file.
        export_dir (str): Directory to save exported models.
        scripted_name (str): Name of the TorchScript file.
        quantized_name (str): Name of the quantized file.

    Returns:
        torch.jit.ScriptModule: Quantized model
    """
    os.makedirs(export_dir, exist_ok=True)

    model = YOLO(pt_path)

    # Export to TorchScript using Ultralytics API
    export_result = model.export(format="torchscript", dynamic=True, simplify=True)

    scripted_path = export_result[0] if isinstance(export_result, (list, tuple)) else export_result
    if not scripted_path.endswith(".pt"):
        raise RuntimeError(f"Unexpected export result path: {scripted_path}")

    # Move exported TorchScript model to target path
    output_scripted_path = os.path.join(export_dir, scripted_name)
    os.rename(scripted_path, output_scripted_path)

    # Load and quantize
    scripted_model = torch.jit.load(output_scripted_path)

    quantized_model = torch.quantization.quantize_dynamic(
        scripted_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    output_quantized_path = os.path.join(export_dir, quantized_name)
    torch.jit.save(quantized_model, output_quantized_path)

    print(f"Scripted model saved to: {output_scripted_path}")
    print(f"Quantized model saved to: {output_quantized_path}")

    return quantized_model
