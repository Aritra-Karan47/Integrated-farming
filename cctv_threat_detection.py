import os
import cv2
import face_recognition
from tkinter import Tk, Label, Button, filedialog, simpledialog
import pygame

# Initialize pygame mixer for alarm
pygame.mixer.init()
ALARM_FILE = "alarm.wav"
if not os.path.exists(ALARM_FILE):
    print("[ERROR] FileNotFoundError: No such file or directory: 'alarm.wav'")
    exit()
alarm_sound = pygame.mixer.Sound(ALARM_FILE)

def play_alarm():
    alarm_sound.play()

# Define known personnel folder
known_personnel_folder = "known_personnel"
known_encodings = []
known_names = []
flag = 0

def load_known_personnel():
    if not os.path.exists(known_personnel_folder):
        os.makedirs(known_personnel_folder)
    for file_name in os.listdir(known_personnel_folder):
        image_path = os.path.join(known_personnel_folder, file_name)
        name = os.path.splitext(file_name)[0]
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append(encoding)
        known_names.append(name)

def match_personnel():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            flag = 1
            if True in matches:
                matched_indices = [index for index, match in enumerate(matches) if match]
                first_match_index = matched_indices[0]
                name = known_names[first_match_index]
                flag = 0
            cv2.rectangle(frame, (left, top), (right, bottom), (237, 255, 32), 2)  # Yellow border
            if flag == 0:
                cv2.putText(frame, f"Authorized: {name}", (left - 40, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (158, 49, 255), 2)  # Purple text
            else:
                cv2.putText(frame, "Intruder Detected", (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 37), 2)  # Yellow text
                play_alarm()  # Trigger alarm for intruders
            match_label.config(text=f"Status: {'Authorized: ' + name if flag == 0 else 'Intruder Detected'}")
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
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(rgb_image)[0]
        name = simpledialog.askstring("Add Personnel", "Enter the name of the personnel:")
        if name:
            known_encodings.append(encoding)
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