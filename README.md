# YOLOv8 Handwritten Digit Detector

A computer vision project for detecting and classifying handwritten digit instances with YOLOv8. The model is trained on the HashiDigits dataset and packaged with a small trained checkpoint plus example images for quick inference.

This project treats digit recognition as an object detection problem rather than only an image classification problem. That means the model predicts both the digit class and its bounding box location, which is useful when an image contains multiple handwritten digits.

## Highlights

- Fine-tunes `YOLOv8n` with transfer learning.
- Detects handwritten digit classes `1` through `8`.
- Uses the HashiDigits YOLOv8 dataset format.
- Trains in staged runs with 416px images, then improves with 640px images and augmentation.
- Includes a trained `best.pt` checkpoint for direct inference.
- Includes small example images for testing without downloading the full dataset.
- Keeps the full dataset, training runs, cache files, and reports out of Git.

## Results

The final validation run reported strong performance on the HashiDigits validation split:

| Metric | Value |
| --- | ---: |
| Precision | 99.7% |
| Recall | 99.8% |
| mAP50 | 99.5% |
| mAP50-95 | 99.1% |

These results are strongest on images that match the training distribution. Real-world handwriting can still be affected by lighting, contrast, stroke thickness, image resolution, and handwriting style.

## Repository Structure

```text
notebooks/
  yolov8_handwritten_digit_detection.ipynb   training, evaluation, inference, and visualization workflow
models/
  best.pt                                    trained YOLOv8 checkpoint
examples/
  hashidigits-sample.jpg                     dataset-style sample input
  multi-digit-sample.jpg                     handwritten multi-digit example
  real-handwriting-7.jpeg                    real handwriting example
  small-handwriting-sample.jpeg              small handwriting example
data.yaml                                    expected YOLO dataset configuration
requirements.txt                            Python dependencies
```

## Dataset

The project uses the HashiDigits dataset from Roboflow Universe:

https://universe.roboflow.com/hashiwokakero-digits/hashidigits/dataset/2

Dataset summary:

- 10,000 labeled images
- 6,999 training images
- 2,001 validation images
- 1,000 test images
- YOLOv8 annotation format
- License: CC BY 4.0

The full dataset is not committed to this repository. To retrain the model, download the dataset and place it under a local `data/` directory using this structure:

```text
data/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

The included `data.yaml` points to that layout.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Quick Inference

Use the included trained checkpoint:

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict(source="examples/real-handwriting-7.jpeg", save=True)
```

Prediction images are saved by Ultralytics under `runs/detect/`.

## Training Workflow

The notebook follows this progression:

1. Load `YOLOv8n` pretrained weights.
2. Train on HashiDigits with image size `416`.
3. Resume training from the latest checkpoint.
4. Continue training for additional epochs.
5. Increase image size to `640` and enable augmentation.
6. Load the best trained checkpoint.
7. Run inference on dataset-style and real handwritten examples.
8. Plot training accuracy using `mAP50`.

## Technical Stack

- Python
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Pandas
- Matplotlib
- Pillow
- Jupyter Notebook

## Notes

This is a portfolio-focused version of the project. The repository includes the notebook, trained checkpoint, dataset configuration, and small inference examples, while large generated artifacts and the full dataset are excluded.
