import cv2
import mediapipe as mp


class HandTracker:
    def __init__(
        self,
        mode: bool = False,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ) -> None:
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.results = None

    def find_hands(self, image, draw: bool = True):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )
        return image

    def find_position(self, image, hand_no: int = 0):
        landmark_list = []
        handedness_label = None

        if not self.results or not self.results.multi_hand_landmarks:
            return landmark_list, handedness_label

        if hand_no >= len(self.results.multi_hand_landmarks):
            return landmark_list, handedness_label

        hand = self.results.multi_hand_landmarks[hand_no]
        h, w, _ = image.shape

        if self.results.multi_handedness and hand_no < len(self.results.multi_handedness):
            handedness_label = self.results.multi_handedness[hand_no].classification[0].label

        for idx, lm in enumerate(hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmark_list.append((idx, cx, cy))

        return landmark_list, handedness_label
