# Vehicle Detection with YOLO11 Small

This project is a web-based application for detecting vehicles—specifically **cars**, **trucks**, and **buses**—in images using a YOLO11 Small model. The app provides a simple interface for uploading images and visualizing detection results, with optional model quantization for faster inference.

---

## Features

- Detects **cars**, **trucks**, and **buses** in uploaded images
- Uses a custom-trained YOLO11 Small model
- Option to use quantized model for improved performance
- Interactive web UI built with [Gradio](https://gradio.app/)
- Displays both detection results and bounding box coordinates

---

## Getting Started
## to get this running either

### 1. Clone the Repository

```
git clone https://github.com/yourusername/vehicle-detection-yolo11.git
cd vehicle-detection-yolo11
```

### 2. Set Up the Environment

Create and activate a virtual environment:

```
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

### 3. Model Files

- best model from training is already in the model directory:
  - `best.pt` (standard model)
  - `best_quantized.pt` (will be generated if not present, and only if user toggles checkbox for quantization)

### 4. Run the Application

```
python app.py
```

The Gradio interface will launch in your browser.

## or try accessing
 https://huggingface.co/spaces/hanaMedhat/vehicleDetection but that one is significantly slower
---

## Usage

1. **Upload an image** containing vehicles (can use the ones in test directory).
2. **Adjust the confidence threshold** as needed.
3. **Toggle quantization** for faster inference (optional). (buggy)
4. **Click "Run Detection"** to see results.

Detected vehicles and their bounding boxes will be displayed.

---

## Model Details
- The training was performed on google colab since we there are GPUs on that platform that
are needed for the training.
- I chose to use a pretrained model and fine tune it because that was more time efficient and more accurate, to get a custom made model to perform as well as something like yolo which is already trained on coco would take huge amounts of effort and time, and given the time constraint that wasnt a good option. 
- Between the yolo models I chose to work with yolo11 which is the latest, it is not that much better than v8 but it is quite faster which came in handy (i had tried training a yolov8 and i noticed that it was slower than yolo11)
- Between the yolo11 options i experimented with the nano and small models give we wanted real time performance.
- The small model was generally better than the nano (experiments in the notebook could be inspected) but not by too much.. the small model becomes a better choice when we can also quantize and improve its performance thus getting the best of both worlds.
- I also experimented with number of samples, trying a 500 per positive class and 400 for all negatives (training) and 100 positive and 80 negative (validation) , then i doubled all for the next experiments. The result was improvement in recall and mAP@50-95 especially for classes with less examples (truck). precision remained stable indicating less false positives, and suggesting that as recall improved with more data, the model learned to detect more true objects without increasing false positives — a sign of better generalization rather than overfitting. At least it isnt generating more wrong boxes. 
- The reason i chose to start with less images in total was because yolo is already trained on coco so it would be overkill to start with a large amount of images and also on a small/nano model
- The stategy for negatives was to use background images and also images that contain other classes but not the positive ones (car, truck, bus)
- The above strategy couldve been improved because the model was missing classes and labeling them as background, or maybe the background was dominating many images..
- Overall the evaluation metrics showed the best model (double images amount + s model)
    -- (yolo11s_2) model had:

        mAP@50 ≈ 0.64

        mAP@50-95 ≈ 0.45

        Recall ≈ 0.59
    -- not too bad for small models?

- Training params were typical , model default since yolo adjusts things automatically, and the epochs i set to 100 with a patience of 10 epochs
- During inference, before I applied any preprocessing of the image it would take longer to predict, but after introducing things like resizing and other operations , the time improved largely
- 
<!--  
Describe your YOLO11 Small model here:
- Training dataset and annotations
- Classes used (car, truck, bus)
- Training parameters and epochs
- Quantization details
- Performance metrics (mAP, inference speed, etc.)
-->

---

## Challenges Faced / Points Still Needing Improvement
- Quantization required converting the model to TorchScript, and certain functions needed adjustments. There are still some issues here

- Gradio layout needed tuning to balance input/output sizes and user experience.

- background overrepresented


## File Structure

```
.
├── app.py
├── model/
│   ├── best.pt
│   └── best_quantized.pt
├── requirements.txt
├── utils.py
├── model.py (contains class that encapsulates model and its functionalities)
├── vehicleDetectionTraining.ipynb
└── ...
```

---

## Acknowledgements

- [YOLO11](https://github.com/ultralytics/yolov5) (adapted for YOLO11 Small)
- [Gradio](https://gradio.app/)

---


