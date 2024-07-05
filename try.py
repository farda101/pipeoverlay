import cv2
import mediapipe as mp
import numpy as np
import os

# Inisialisasi MediaPipe dan OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Fungsi untuk memuat semua gambar baju dari folder
def load_clothes_from_folder(folder_path):
    clothes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # Pastikan hanya memuat file PNG
            clothes.append(cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_UNCHANGED))
    return clothes

# Path ke folder yang berisi gambar baju
clothes_folder = "Resources/Shirts"  # Ganti dengan path ke folder gambar baju Anda
clothes = load_clothes_from_folder(clothes_folder)

if not clothes:
    raise ValueError("Tidak ada gambar baju yang ditemukan di folder 'clothes'.")

current_clothes_index = 0

def overlay_clothes(image, resized_clothes, top_left_x, top_left_y):
    for i in range(resized_clothes.shape[0]):
        for j in range(resized_clothes.shape[1]):
            y, x = top_left_y + i, top_left_x + j
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                if resized_clothes[i, j, 3] != 0:
                    image[y, x, :] = resized_clothes[i, j, :3]

def main():
    global current_clothes_index
    
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Convert the frame back to BGR
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # Check if any pose landmarks are detected
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates of shoulders and hips for better positioning
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                # Calculate the center and size of the overlay area
                center_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
                shoulder_width = int(abs(left_shoulder.x - right_shoulder.x) * frame.shape[1])
                torso_height = int(abs(left_shoulder.y - left_hip.y) * frame.shape[0])

                # Define the width and height of the clothes
                width = shoulder_width
                height = torso_height

                # Ensure width and height are greater than zero
                if width > 0 and height > 0:
                    # Resize the current clothes image
                    resized_clothes = cv2.resize(clothes[current_clothes_index], (width, height))

                    # Calculate the top-left corner of the overlay area
                    top_left_x = center_x - width // 2
                    top_left_y = int(left_shoulder.y * frame.shape[0]) - height // 3  # Adjusting to fit better

                    # Ensure the overlay coordinates are within the frame boundaries
                    if top_left_x < 0:
                        resized_clothes = resized_clothes[:, -top_left_x:]
                        top_left_x = 0
                    if top_left_y < 0:
                        resized_clothes = resized_clothes[-top_left_y:, :]
                        top_left_y = 0
                    if top_left_x + resized_clothes.shape[1] > frame.shape[1]:
                        resized_clothes = resized_clothes[:, :frame.shape[1] - top_left_x]
                    if top_left_y + resized_clothes.shape[0] > frame.shape[0]:
                        resized_clothes = resized_clothes[:frame.shape[0] - top_left_y, :]

                    # Overlay the clothes on the frame
                    overlay_clothes(frame, resized_clothes, top_left_x, top_left_y)

            # Draw buttons
            frame = cv2.rectangle(frame, (50, 50), (150, 150), (255, 0, 0), -1)
            frame = cv2.rectangle(frame, (500, 50), (600, 150), (0, 255, 0), -1)
            cv2.putText(frame, 'Prev', (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, 'Next', (510, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Virtual Try-On', frame)

            # Check for mouse click to change clothes
            def mouse_click(event, x, y, flags, param):
                global current_clothes_index
                if event == cv2.EVENT_LBUTTONDOWN:
                    if 50 <= x <= 150 and 50 <= y <= 150:
                        current_clothes_index = (current_clothes_index - 1) % len(clothes)
                    elif 500 <= x <= 600 and 50 <= y <= 150:
                        current_clothes_index = (current_clothes_index + 1) % len(clothes)

            cv2.setMouseCallback('Virtual Try-On', mouse_click)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
