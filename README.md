# AI-Based Personal Item Recognition System

## Project Overview

This project is part of a university team project focused on developing an AI-powered home assistant.

The current prototype is a Python-based computer vision application that uses a computer webcam to detect objects and recognize registered personal belongings through visual feature matching.

The system combines YOLO-based object detection with ORB feature matching and geometric verification to improve personal item recognition.

## My Contributions

I developed the current computer vision application, including:

- Real-time webcam video processing using OpenCV.
- Object detection using YOLO11n.
- Personal item registration using multiple reference images.
- Image preprocessing and object cropping.
- ORB feature extraction and feature matching.
- Geometric verification for visual similarity.
- Multi-frame verification before confirming an item match.
- Streamlit-based user interface.
- SQLite database integration for storing registered items and detection history.

ESP32-CAM image stream integration is a planned development task that I am responsible for.

## Technologies

- Python
- OpenCV
- YOLO11n
- Ultralytics
- ORB Feature Matching
- Streamlit
- SQLite
- NumPy

## Implemented Features

- Real-time computer webcam input.
- YOLO-based object detection.
- Personal item registration with multiple reference images.
- Visual feature matching using ORB.
- Geometric verification of candidate matches.
- Multi-frame verification.
- Local database for registered items and detection history.
- Interactive Streamlit interface.

## Development Status

| Component | Status |
|---|---|
| Webcam-based video processing | Implemented |
| YOLO object detection | Implemented |
| Personal item registration | Implemented |
| ORB feature matching | Implemented |
| Geometric verification | Implemented |
| Multi-frame verification | Implemented |
| Streamlit interface | Implemented |
| SQLite database | Implemented |
| ESP32-CAM integration | Planned / In development |
| IP camera processing | Developed separately by another team member |

The current prototype uses a computer webcam. ESP32-CAM and IP camera integration are not yet part of the current working prototype.

## How It Works

1. The user registers a personal item by uploading multiple reference images.
2. The application processes the incoming webcam frames.
3. YOLO detects objects in the video stream.
4. ORB extracts and compares visual features between detected objects and registered reference images.
5. Geometric verification helps evaluate candidate matches.
6. Multiple frames are evaluated before the application confirms a potential match.
7. Registered items and detection history are stored locally using SQLite.

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The YOLO11n model weights are downloaded automatically by Ultralytics if they are not already available locally.

## Privacy and Data

Personal reference images and local database files are excluded from the repository.

Registered item images and detection history are stored locally.

## Project Context

This is a university team project. My individual contribution focuses on developing the computer vision application and working toward its integration with ESP32-CAM hardware.
