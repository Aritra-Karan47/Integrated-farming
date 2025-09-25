import os
import cv2
import numpy as np
import pygame
from tkinter import Tk, Label, Button, filedialog, simpledialog
from ultralytics import YOLO  # For YOLOv8
from facenet_pytorch import MTCNN, InceptionResnetV1  # For deep learning face detection and recognition
import torch

# Initialize pygame mixer for alarm
pygame.mixer.init()
ALARM_FILE = "alarm.wav"
if not os.path.exists(ALARM_FILE):
    print("[ERROR] FileNotFoundError: No such file or directory: 'alarm.wav'")
    exit()
alarm_sound = pygame.mixer.Sound(ALARM_FILE)

def play_alarm():
    alarm_sound.play()

# Load YOLOv8 model for people detection
model = YOLO("yolov8n.pt")

# Load facenet models (deep learning)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)  # Face detector
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Embedding extractor

# Define known personnel folder and load known embeddings
known_personnel_folder = "known_personnel"
known_embeddings = []  # Store deep learning embeddings
known_names = []

def load_known_personnel():
    if not os.path.exists(known_personnel_folder):
        os.makedirs(known_personnel_folder)
    for file_name in os.listdir(known_personnel_folder):
        image_path = os.path.join(known_personnel_folder, file_name)
        name = os.path.splitext(file_name)[0]
        image = cv2.imread(image_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces, _ = mtcnn.detect(image_rgb)  # Detect faces
            if faces is not None and len(faces) > 0:
                aligned = mtcnn(image_rgb)  # Align and crop
                if aligned is not None:
                    embedding = resnet(aligned).detach().cpu().numpy()[0]  # Get embedding
                    known_embeddings.append(embedding)
                    known_names.append(name)

def match_personnel():
    cap = cv2.VideoCapture(0)
    people_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        # People counting with YOLOv8
        results = model(frame)
        people_count = sum(1 for result in results[0].boxes if result.cls == 0)  # Class 0 is person
        for result in results[0].boxes:
            if result.cls == 0:  # Person class
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for people

        # Face detection and recognition with facenet-pytorch
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, _ = mtcnn.detect(frame_rgb)  # Detect faces
        intruder_detected = False
        if faces is not None:
            aligned = mtcnn(frame_rgb)  # Align and crop detected faces
            if aligned is not None:
                embeddings = resnet(aligned).detach().cpu().numpy()  # Get embeddings
                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = map(int, face)
                    embedding = embeddings[i]
                    min_dist = float('inf')
                    name = "Unknown"
                    for known_embedding, known_name in zip(known_embeddings, known_names):
                        dist = np.linalg.norm(known_embedding - embedding)  # Euclidean distance
                        if dist < min_dist:
                            min_dist = dist
                            name = known_name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (237, 255, 32), 2)  # Yellow border
                    if min_dist < 1.0:  # Threshold for match (adjust 0.8-1.2 based on testing)
                        cv2.putText(frame, f"Authorized: {name}", (x1 - 40, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (158, 49, 255), 2)
                    else:
                        cv2.putText(frame, "Intruder Detected", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 37), 2)
                        intruder_detected = True
                        play_alarm()  # Trigger alarm for intruders

        match_label.config(text=f"Status: {people_count} people, {'Intruder Detected' if intruder_detected else 'All Authorized'}")
        cv2.imshow('Hydroponics Security', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def add_personnel():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                           filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")))
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces, _ = mtcnn.detect(image_rgb)
            if faces is not None and len(faces) > 0:
                aligned = mtcnn(image_rgb)
                if aligned is not None:
                    embedding = resnet(aligned).detach().cpu().numpy()[0]
                    name = simpledialog.askstring("Add Personnel", "Enter the name of the personnel:")
                    if name:
                        known_embeddings.append(embedding)
                        known_names.append(name)
                        file_name = name + ".jpg"
                        save_path = os.path.join(known_personnel_folder, file_name)
                        cv2.imwrite(save_path, image)
                        print("Personnel added successfully.")

load_known_personnel()

root = Tk()
root.title("Personnel Identification System")
root.geometry("500x300")

label = Label(root, text="Welcome to the Hydroponics Personnel Identification System")
label.pack()

match_button = Button(root, text="Match Personnel", command=match_personnel)
match_button.pack()
add_button = Button(root, text="Add Personnel", command=add_personnel)
add_button.pack()
match_label = Label(root, text="Status: ")
match_label.pack()
root.mainloop()