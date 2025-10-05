## Real-Time Facial Emotion Recognition (ITS Computer Vision Course)

A compact, course-oriented project that uses a computer's webcam to detect a face and recognize the person's emotion in real time. Built for the ITS Computer Vision class.

### Key Features
- Real-time webcam capture
- Face detection
- Emotion classification over common categories (e.g., happy, sad, neutral, surprise, anger, fear, disgust)
- On-screen visualization (bounding box, label, confidence)

### Project Goals
- Demonstrate a complete computer vision inference loop from camera input to structured output
- Compare different face detectors and emotion classifiers (optional)

### High-Level Architecture
1. Video source: laptop/PC webcam
2. Face detection: locate faces in each frame
3. Preprocessing: crop/align, normalize, resize
4. Emotion model: deep classifier outputs probabilities
5. Post-processing: smooth predictions over time (optional)
6. Visualization: draw boxes, labels, FPS overlay

### Tech Stack (suggested)
- Python 3.9+
- OpenCV for video I/O and basic CV ops
- PyTorch or TensorFlow/Keras for the emotion model
- Numpy/Scipy for preprocessing utilities

### Datasets and Models
- FER2013

### Performance Expectations
- Target ≥ 20 FPS on a mid-range laptop with a lightweight model
- Prediction smoothing (temporal averaging) recommended to reduce flicker

### Ethical Use and Limitations
- Emotion recognition is probabilistic and may be biased by lighting, pose, occlusions, and dataset bias
- Do not use for high-stakes decisions; this is an educational project
- Obtain consent before recording or analyzing video of others

### Repository Structure (proposed)
```
Emotional_recognizer_ITS_CV/
  ├─ src/
  │  ├─ camera.py             # webcam capture
  │  ├─ detector.py           # face detection wrapper
  │  ├─ preprocessing.py      # crop/align/normalize
  │  ├─ model.py              # load/run emotion model
  │  ├─ visualizer.py         # drawing overlays
  │  └─ app.py                # main loop (entry point)
  ├─ models/                  # pretrained weights, model cards
  ├─ configs/                 # yaml/json configs
  ├─ notebooks/               # exploration/training
  ├─ data/                    # sample images (no PII)
  ├─ requirements.txt         # Python dependencies
  └─ README.md
```

### Installation
We will finalize step-by-step setup with you. Suggested baseline:
1. Create and activate a Python virtual environment
2. Install dependencies from `requirements.txt`
3. Download/place the pretrained emotion model in `models/`
4. Run the app and allow webcam access

### Quick Start (to be finalized)
```bash
# 1) Create venv (example)
python -m venv .venv && .venv\\Scripts\\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Launch the real-time demo
python -m src.app --camera 0 --config configs/default.yaml
```

### Configuration
- `configs/default.yaml` controls detector type, model path, input size, labels, smoothing, thresholds, and visualization options

### Roadmap
1) Use webcam to extract face
2) Convert the face image to an image that looks like those in the dataset
3) Train the model based on the FER2013 dataset
4) Pass the webcam image in the model
5) Extract the result and add it on the face box (visible on the webcam return)

### License
Specify your license here (e.g., MIT). Make sure any pretrained model you use permits redistribution.

### Acknowledgements
- ITS Computer Vision course staff and materials
- Authors and maintainers of datasets and pretrained models used
