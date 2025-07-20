import cv2, mediapipe as mp, pyautogui, math
pyautogui.FAILSAFE = False  # Prevents sudden failsafe errors

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Utility: Euclidean distance
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# Keep track for scroll gesture
prev_y = None

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Map finger tips
        idx = lm[8]  # index tip
        mid = lm[12]  # middle tip
        thumb = lm[4]  # thumb tip

        cx, cy = int(idx.x * w), int(idx.y * h)
        sx, sy = int(idx.x * screen_w), int(idx.y * screen_h)
        pyautogui.moveTo(sx, sy, duration=0)

        d_thumb = distance(idx, thumb)
        d_mid = distance(idx, mid)

        # Left-Click: Index + Thumb
        if d_thumb < 0.05:
            pyautogui.click()
            cv2.putText(frame, "Left Click", (10, 40), 1, 1, (0, 255, 0), 2)

        # Right-Click: Index + Middle
        elif d_mid < 0.05:
            pyautogui.click(button='right')
            cv2.putText(frame, "Right Click", (10, 40), 1, 1, (255, 0, 0), 2)

        # Scrolling: Index + Middle far apart
        else:
            if prev_y:
                if cy < prev_y - 20:
                    pyautogui.scroll(40)
                    cv2.putText(frame, "Scroll Up", (10, 40), 1, 1, (0, 255, 255), 2)
                elif cy > prev_y + 20:
                    pyautogui.scroll(-40)
                    cv2.putText(frame, "Scroll Down", (10, 40), 1, 1, (0, 255, 255), 2)
            prev_y = cy

    cv2.imshow("Gesture Mouse", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
