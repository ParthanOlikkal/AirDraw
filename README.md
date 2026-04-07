
AirDraw is a real-time virtual drawing application built with **MediaPipe Hands** and **OpenCV**. It uses webcam input to detect hand landmarks and converts the movement of your index finger into brush strokes on a digital canvas.

## Features

- Real-time hand tracking with MediaPipe
- Draw using your **index finger**
- Use **index + middle finger** to move without drawing
- Pick colors from the on-screen toolbar
- Eraser mode
- Pinch gesture to clear the canvas
- Save drawings as PNG images
- Simple smoothing to reduce jitter

## Controls

- **Index finger up**: draw
- **Index + middle fingers up**: selection / move mode
- **Touch toolbar with index finger**: change color or select eraser
- **Thumb + index pinch**: clear canvas
- **s**: save current canvas
- **q**: quit

## Project Structure

```text
airdraw_project/
├── main.py
├── hand_tracker.py
├── gesture_utils.py
├── requirements.txt
├── README.md
└── outputs/
```

## Installation

1. Create and activate a virtual environment (recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```


## Next Improvements

- Brush thickness control using pinch distance
- Whiteboard-only mode
- Shape drawing mode
- Multi-hand support
- Stroke undo/redo
- OCR for handwritten notes
