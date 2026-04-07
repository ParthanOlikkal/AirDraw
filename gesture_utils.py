import math
from typing import List, Optional, Tuple

Landmark = Tuple[int, int, int]
Point = Tuple[int, int]

FINGER_TIPS = [4, 8, 12, 16, 20]


def fingers_up(landmarks: List[Landmark], handedness: Optional[str] = None) -> List[int]:
    """
    Returns [thumb, index, middle, ring, pinky].
    Works reasonably well for a single mirrored webcam feed.
    """
    if len(landmarks) < 21:
        return [0, 0, 0, 0, 0]

    fingers = []

    thumb_tip_x = landmarks[FINGER_TIPS[0]][1]
    thumb_joint_x = landmarks[FINGER_TIPS[0] - 1][1]

    if handedness == "Right":
        fingers.append(1 if thumb_tip_x > thumb_joint_x else 0)
    elif handedness == "Left":
        fingers.append(1 if thumb_tip_x < thumb_joint_x else 0)
    else:
        fingers.append(1 if abs(thumb_tip_x - thumb_joint_x) > 10 else 0)

    for tip_id in FINGER_TIPS[1:]:
        fingers.append(1 if landmarks[tip_id][2] < landmarks[tip_id - 2][2] else 0)

    return fingers


def get_point(landmarks: List[Landmark], idx: int) -> Optional[Point]:
    for lm_id, x, y in landmarks:
        if lm_id == idx:
            return (x, y)
    return None


def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def smooth_point(prev_point: Optional[Point], new_point: Point, alpha: float = 0.35) -> Point:
    if prev_point is None:
        return new_point
    x = int((1 - alpha) * prev_point[0] + alpha * new_point[0])
    y = int((1 - alpha) * prev_point[1] + alpha * new_point[1])
    return x, y
