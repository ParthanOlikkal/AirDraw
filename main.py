import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from gesture_utils import distance, fingers_up, get_point, smooth_point
from hand_tracker import HandTracker


Point = Tuple[int, int]

WINDOW_NAME = "AirDraw"
OUTPUT_DIR = "outputs"
WIDTH = 1700
HEIGHT = 720
HEADER_HEIGHT = 90

COLORS = {
    "Magenta": (255, 0, 255),
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Eraser": (0, 0, 0),
}

TOOLBAR = [
    ("Magenta", (20, 15, 150, 70)),
    ("Blue", (170, 15, 300, 70)),
    ("Green", (320, 15, 450, 70)),
    ("Red", (470, 15, 600, 70)),
    ("Eraser", (620, 15, 780, 70)),
]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


class AirDrawApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        self.tracker = HandTracker(max_num_hands=1)
        self.canvas = None
        self.prev_draw_point: Optional[Point] = None
        self.smoothed_point: Optional[Point] = None

        self.current_color_name = "Magenta"
        self.brush_thickness = 7
        self.eraser_thickness = 40
        self.mode_text = "Idle"

        self.last_clear_time = 0.0
        self.last_save_time = 0.0

    def create_canvas_if_needed(self, frame):
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)

    def draw_toolbar(self, frame):
        cv2.rectangle(frame, (0, 0), (WIDTH, HEADER_HEIGHT), (35, 35, 35), -1)
        for name, (x1, y1, x2, y2) in TOOLBAR:
            color = COLORS[name] if name != "Eraser" else (220, 220, 220)
            thickness = -1 if self.current_color_name == name else 2
            fill_color = color if self.current_color_name == name else (70, 70, 70)
            cv2.rectangle(frame, (x1, y1), (x2, y2), fill_color, thickness)
            text_color = (0, 0, 0) if self.current_color_name == name and name != "Eraser" else (255, 255, 255)
            cv2.putText(frame, name, (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        cv2.putText(frame, f"Mode: {self.mode_text}", (820, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Draw: index | Select: index+middle | Clear: pinch | Save: s | Quit: q", (820, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)

    def handle_toolbar_selection(self, point: Point):
        x, y = point
        if y > HEADER_HEIGHT:
            return False
        for name, (x1, y1, x2, y2) in TOOLBAR:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.current_color_name = name
                self.mode_text = f"Selected {name}"
                return True
        return False

    def draw_line(self, start: Point, end: Point):
        color = COLORS[self.current_color_name]
        thickness = self.eraser_thickness if self.current_color_name == "Eraser" else self.brush_thickness
        cv2.line(self.canvas, start, end, color, thickness)

    def clear_canvas(self):
        self.canvas[:] = 0
        self.prev_draw_point = None
        self.smoothed_point = None
        self.mode_text = "Clear"

    def save_canvas(self) :
        ensure_output_dir()
        ts = int(time.time())
        canvas_path = os.path.join(OUTPUT_DIR, f"airdraw_canvas_{ts}.png")
        cv2.imwrite(canvas_path, self.canvas)
        self.last_save_time = time.time()
        return canvas_path

    def blend_frame_and_canvas(self, frame):
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        merged = cv2.bitwise_and(frame, inv)
        merged = cv2.bitwise_or(merged, self.canvas)
        return merged

    def run(self):
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        while True:
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            self.create_canvas_if_needed(frame)

            frame = self.tracker.find_hands(frame, draw=True)
            landmarks, handedness = self.tracker.find_position(frame)
            self.mode_text = "Idle"

            if landmarks:
                index_tip = get_point(landmarks, 8)
                middle_tip = get_point(landmarks, 12)
                thumb_tip = get_point(landmarks, 4)
                finger_state = fingers_up(landmarks, handedness)

                if index_tip:
                    self.smoothed_point = smooth_point(self.smoothed_point, index_tip, alpha=0.35)
                    ix, iy = self.smoothed_point
                    cv2.circle(frame, (ix, iy), 10, (0, 255, 255), cv2.FILLED)

                    if finger_state[1] == 1 and finger_state[2] == 1:
                        self.prev_draw_point = None
                        if self.handle_toolbar_selection((ix, iy)):
                            pass
                        else:
                            self.mode_text = "Move"
                            if middle_tip:
                                mx, my = middle_tip
                                cv2.rectangle(frame, (min(ix, mx) - 20, min(iy, my) - 20), (max(ix, mx) + 20, max(iy, my) + 20), (0, 255, 0), 2)

                    elif finger_state[1] == 1 and finger_state[2] == 0:
                        if iy < HEADER_HEIGHT and self.handle_toolbar_selection((ix, iy)):
                            self.prev_draw_point = None
                        else:
                            self.mode_text = "Erase" if self.current_color_name == "Eraser" else "Draw"
                            current_point = (ix, iy)
                            if self.prev_draw_point is None:
                                self.prev_draw_point = current_point
                            self.draw_line(self.prev_draw_point, current_point)
                            self.prev_draw_point = current_point
                    else:
                        self.prev_draw_point = None

                if thumb_tip and index_tip:
                    pinch_dist = distance(thumb_tip, index_tip)
                    if pinch_dist < 35 and (time.time() - self.last_clear_time) > 1.0:
                        self.clear_canvas()
                        self.last_clear_time = time.time()

            else:
                self.prev_draw_point = None
                self.smoothed_point = None

            output = self.blend_frame_and_canvas(frame)
            self.draw_toolbar(output)
            cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and (time.time() - self.last_save_time) > 0.5:
                path = self.save_canvas()
                print(f"Saved canvas to: {path}")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    AirDrawApp().run()