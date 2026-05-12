import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Audio setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lm_list = []

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand_landmark,
                mp_hands.HAND_CONNECTIONS
            )

            for id, lm in enumerate(hand_landmark.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

        if lm_list:
            # Thumb tip
            x1, y1 = lm_list[4][1], lm_list[4][2]

            # Index finger tip
            x2, y2 = lm_list[8][1], lm_list[8][2]

            # Draw circles
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)

            # Draw line
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Distance between fingers
            length = hypot(x2 - x1, y2 - y1)

            # Convert distance to volume
            vol = np.interp(length, [20, 200], [min_vol, max_vol])

            volume.SetMasterVolumeLevel(vol, None)

            # Volume bar
            vol_bar = np.interp(length, [20, 200], [400, 150])
            vol_percent = np.interp(length, [20, 200], [0, 100])

            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(
                img,
                (50, int(vol_bar)),
                (85, 400),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                img,
                f'{int(vol_percent)} %',
                (40, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3
            )

    cv2.imshow("AI Hand Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
