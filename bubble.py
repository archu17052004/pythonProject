import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Function to detect hand landmarks using MediaPipe
def detect_hand_landmarks(image, mp_hands, draw=True):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if draw:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))
    return image, landmarks

# Function to check if a point is inside a circle
def is_inside_circle(x, y, circle_center, radius):
    return (x - circle_center[0])**2 + (y - circle_center[1])**2 <= radius**2

# Function to generate random circle coordinates
def generate_circle_coordinates(width, height, radius):
    return (random.randint(radius, width - radius), random.randint(radius, height - radius))

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Create a video capture object
cap = cv2.VideoCapture(0)

# Set the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the radius of the bubble
bubble_radius = 30

# Set the score
score = 0

# Set the interval between bubble appearances (in milliseconds)
appearance_interval = 1000 // 15  # 15 bubbles per second

# Track the time of last bubble appearance
last_appearance_time = time.time()

start_window_width=800
start_window_height=600

# Increase display size and slow down bubble movement
cv2.namedWindow('Bubble Popping Game', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Bubble Popping Game', 1280, 720)  # Set the window size as needed
bubble_speed = 1  # Adjust bubble speed as needed

# Initialize circle_center
circle_center = generate_circle_coordinates(frame_width, frame_height, bubble_radius)

background_image = cv2.imread('C:/Users/HP/Downloads/ocean.jpg')
background_image = cv2.resize(background_image, (start_window_width, start_window_height))

# Function to handle mouse click events
def on_mouse_click(event, x, y, flags, param):
    global game_started
    if event == cv2.EVENT_LBUTTONDOWN:
        game_started = True

# Create a window to start the game
start_window = np.zeros((300, 600, 3), dtype=np.uint8)
cv2.putText(background_image, 'Click to Start Game', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
cv2.imshow('Start Game', background_image)
text_position=(int(start_window_width*0.3),int(start_window_height*0.5))
cv2.setMouseCallback('Start Game', on_mouse_click)

# Wait for the user to start the game
game_started = False
while not game_started:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow('Start Game')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hand landmarks
    frame, landmarks = detect_hand_landmarks(frame, mp_hands)

    # Generate a random circle after the appearance interval
    current_time = time.time()
    if (current_time - last_appearance_time) * 1000 > appearance_interval:
        circle_center = generate_circle_coordinates(frame_width, frame_height, bubble_radius)
        last_appearance_time = current_time

    # Draw the circle
    cv2.circle(frame, circle_center, bubble_radius, (255, 0, 0), -1)

    # Check if any hand landmark is inside the bubble
    for landmark in landmarks:
        if is_inside_circle(landmark[0], landmark[1], circle_center, bubble_radius):
            score += 1
            circle_center = generate_circle_coordinates(frame_width, frame_height, bubble_radius)
            cv2.circle(frame, circle_center, bubble_radius, (255, 0, 0), -1)

    # Move bubbles upwards
    if circle_center[1] > 0:
        circle_center = (circle_center[0], circle_center[1] - bubble_speed)

    # Display the score
    cv2.putText(frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Bubble Popping Game', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()