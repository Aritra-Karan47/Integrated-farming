from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import requests
from twilio.rest import Client
from typing import List, Optional
import cv2
import numpy as np
import pygame
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import time
import re
import easyocr
import pandas as pd
import io
import base64
from pytz import timezone as pytz_timezone
import random
import json

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Neon DB setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Twilio setup
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_CLIENT = Client(TWILIO_SID, TWILIO_TOKEN)
TWILIO_SERVICE_SID = 'MG2eb30cc4e79581a7cbfb3fdda0d35fdc'

# Auth setup
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    phone_number = Column(String, unique=True)

class HydroponicsData(Base):
    __tablename__ = "hydroponics_data"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now())
    water_level = Column(Float, nullable=True)
    tray_count = Column(Integer, nullable=True)
    humidity = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    ph = Column(Float, nullable=True)

class Todo(Base):
    __tablename__ = "todos"
    id = Column(Integer, primary_key=True, index=True)
    task = Column(String)
    is_done = Column(Boolean, default=False)
    user_id = Column(Integer)
    last_reset = Column(DateTime, default=func.now())

Base.metadata.create_all(bind=engine)

# Pydantic Schemas
class UserCreate(BaseModel):
    username: str
    password: str
    phone_number: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class SensorData(BaseModel):
    timestamp: datetime
    water_level: Optional[float]
    tray_count: Optional[int]
    humidity: Optional[float]
    temperature: Optional[float]
    ph: Optional[float]

class TodoCreate(BaseModel):
    task: str

class TodoOut(BaseModel):
    id: int
    task: str
    is_done: bool

# Auth helpers
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        token_data = TokenData(username=username)
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    db = SessionLocal()
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    return user

# Geolocation helper
def get_client_city(request: Request):
    client_ip = request.client.host
    if client_ip == "127.0.0.1":
        return "Mumbai"
    try:
        api_key = os.getenv("IPINFO_API_KEY")
        url = f"https://ipinfo.io/{client_ip}/json?token={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("city", "Mumbai")
        return "Mumbai"
    except Exception:
        return "Mumbai"

# Weather helper
def get_weather(city):
    api_key = os.getenv("OPEN_WEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {'temp': data['main']['temp'], 'description': data['weather'][0]['description']}
    return {'temp': 'N/A', 'description': 'N/A'}

# Initialize fixed tasks for a user
def initialize_fixed_tasks(user_id):
    db = SessionLocal()
    fixed_tasks = [
        "Check and adjust pH",
        "Monitor nutrient solution levels",
        "Inspect roots and plants",
        "Clean pumps and pipes",
        "Record growth and health"
    ]
    existing_tasks = db.query(Todo).filter(Todo.user_id == user_id).all()
    existing_task_names = {task.task for task in existing_tasks}
    for task in fixed_tasks:
        if task not in existing_task_names:
            new_todo = Todo(task=task, user_id=user_id)
            db.add(new_todo)
    db.commit()

# Reset tasks daily
def reset_daily_tasks(user_id):
    db = SessionLocal()
    ist = pytz_timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    tasks = db.query(Todo).filter(Todo.user_id == user_id).all()
    for task in tasks:
        time_diff = current_time - task.last_reset.replace(tzinfo=ist) if task.last_reset else timedelta(days=2)
        if time_diff.total_seconds() >= 24 * 3600:
            task.is_done = False
            task.last_reset = current_time
    db.commit()

# Populate sensor data
def populate_sensor_data():
    db = SessionLocal()
    ist = pytz_timezone('Asia/Kolkata')
    start_time = datetime.now(ist) - timedelta(days=7)  # Start 7 days ago
    for _ in range(100):
        timestamp = start_time + timedelta(minutes=random.randint(0, 10080))  # 7 days = 10,080 minutes
        water_level = round(random.uniform(5.0, 15.0), 2)  # 5-15 cm
        tray_count = random.randint(1, 10)  # 1-10 trays
        humidity = round(random.uniform(40.0, 80.0), 2)  # 40-80%
        temperature = round(random.uniform(18.0, 28.0), 2)  # 18-28°C
        ph = round(random.uniform(5.5, 6.5), 2)  # 5.5-6.5 pH
        sensor_data = HydroponicsData(
            timestamp=timestamp,
            water_level=water_level,
            tray_count=tray_count,
            humidity=humidity,
            temperature=temperature,
            ph=ph
        )
        db.add(sensor_data)
    db.commit()

@app.get("/seed-sensor-data")
async def seed_sensor_data(current_user: User = Depends(get_current_user)):
    populate_sensor_data()
    return {"message": "Sensor data seeded with 100 records"}

# Routes
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), phone_number: str = Form(...)):
    db = SessionLocal()
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed_password, phone_number=phone_number)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/token", response_model=Token)
async def login_for_access_token(username: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="access_token", value=access_token)
    return response

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: User = Depends(get_current_user)):
    view = request.query_params.get("view", "visualization")  # Default to visualization
    city = get_client_city(request)
    weather = get_weather(city)
    db = SessionLocal()
    data = db.query(HydroponicsData).order_by(HydroponicsData.timestamp.desc()).limit(100).all()  # Fetch last 100 records
    sensor_data = [
        {
            "timestamp": d.timestamp.strftime("%Y-%m-%d %H:%M:%S") if d.timestamp else "",
            "water_level": d.water_level,
            "tray_count": d.tray_count,
            "humidity": d.humidity,
            "temperature": d.temperature,
            "ph": d.ph
        }
        for d in data
    ]
    todos = db.query(Todo).filter(Todo.user_id == current_user.id).all()
    if not todos:
        initialize_fixed_tasks(current_user.id)
        todos = db.query(Todo).filter(Todo.user_id == current_user.id).all()
    reset_daily_tasks(current_user.id)
    todos = db.query(Todo).filter(Todo.user_id == current_user.id).all()
    return templates.TemplateResponse("dashboard.html", {"request": request, "view": view, "weather": weather, "sensor_data": sensor_data, "todos": todos})

@app.post("/todos/{todo_id}/done")
async def mark_todo_done(todo_id: int, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    todo = db.query(Todo).filter(Todo.id == todo_id, Todo.user_id == current_user.id).first()
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    todo.is_done = True
    todo.last_reset = datetime.now(timezone.utc)
    db.commit()
    background_tasks.add_task(send_sms_notification, current_user.phone_number, todo.task)
    return {"message": "Todo marked done, notification sent"}

def send_sms_notification(phone_number: str, task: str):
    message = TWILIO_CLIENT.messages.create(
        messaging_service_sid=TWILIO_SERVICE_SID,
        body=f"Task '{task}' marked done.",
        to=phone_number
    )
    print(message.sid)

@app.get("/cctv_human_detection", response_class=HTMLResponse)
async def cctv_human_detection(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("cctv_human_detection.html", {"request": request})

@app.get("/cctv_stream")
async def cctv_stream():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    pygame.mixer.init()
    ALARM_FILE = "alarm.wav"
    if not os.path.exists(ALARM_FILE):
        print("[ERROR] FileNotFoundError: No such file or directory: 'alarm.wav'")
        return

    alarm_sound = pygame.mixer.Sound(ALARM_FILE)
    def play_alarm():
        alarm_sound.play()

    model = YOLO("yolov8n.pt")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    known_personnel_folder = "known_personnel"
    known_embeddings = []
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
                faces, _ = mtcnn.detect(image_rgb)
                if faces is not None and len(faces) > 0:
                    aligned = mtcnn(image_rgb)
                    if aligned is not None:
                        embedding = resnet(aligned).detach().cpu().numpy()[0]
                        known_embeddings.append(embedding)
                        known_names.append(name)

    load_known_personnel()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        people_count = sum(1 for result in results[0].boxes if result.cls == 0)
        for result in results[0].boxes:
            if result.cls == 0:  # Class 0 is typically 'person' in COCO dataset
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add FaceNet logic to frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, _ = mtcnn.detect(frame_rgb)
        intruder_detected = False
        if faces is not None:
            aligned = mtcnn(frame_rgb)
            if aligned is not None:
                embeddings = resnet(aligned).detach().cpu().numpy()
                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = map(int, face)
                    embedding = embeddings[i]
                    min_dist = float('inf')
                    name = "Unknown"
                    for known_embedding, known_name in zip(known_embeddings, known_names):
                        dist = np.linalg.norm(known_embedding - embedding)
                        if dist < min_dist:
                            min_dist = dist
                            name = known_name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (237, 255, 32), 2)
                    if min_dist < 1.0:
                        cv2.putText(frame, f"Authorized: {name}", (x1 - 40, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (158, 49, 255), 2)
                    else:
                        cv2.putText(frame, "Intruder Detected", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 37), 2)
                        intruder_detected = True
                        play_alarm()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.get("/motor_control", response_class=HTMLResponse)
async def motor_control(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("motor_control.html", {"request": request})

@app.post("/motor/automated")
async def automated_motor(length: float, direction: str, current_user: User = Depends(get_current_user)):
    speed = 1.0
    time_to_cover = length / speed
    # Deactivate forward and backward buttons temporarily
    # (Logic to disable buttons should be handled in the frontend)
    time.sleep(time_to_cover)
    return {"message": f"Motor moved {direction} for {length} units"}

@app.post("/motor/manual")
async def manual_motor(direction: str, length: float, current_user: User = Depends(get_current_user)):
    speed = 1.0
    time_to_cover = length / speed
    # Move pointer in the given direction
    time.sleep(time_to_cover)
    return {"message": f"Motor moved {direction} for {length} units", "direction": direction}

@app.get("/tray_count", response_class=HTMLResponse)
async def tray_count_view(request: Request, current_user: User = Depends(get_current_user)):
    digits = run_tray_count_detection()
    return templates.TemplateResponse("tray_count.html", {"request": request, "tray_count": digits})

def run_tray_count_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: cannot open camera"

    reader = easyocr.Reader(['en'], gpu=False)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return "Failed to grab frame"

    raw_text, digits = ocr_with_easyocr(frame, reader)
    cap.release()
    return digits if digits else "No digits found"

def ocr_with_easyocr(frame, reader):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        results = reader.readtext(img_rgb, detail=0)
    except Exception as e:
        print("EasyOCR error:", e)
        return None, None
    joined = " ".join(results).strip()
    digits = extract_digits(joined)
    return joined, digits

def extract_digits(text):
    if not text:
        return None
    groups = re.findall(r'\d+', text)
    return groups[0] if groups else None

@app.post("/sensors/on_off")
async def sensor_toggle(sensor: str, state: bool, current_user: User = Depends(get_current_user)):
    # Add logic to handle sensor state changes (e.g., update database or hardware)
    return {"message": f"{sensor} turned {'on' if state else 'off'}"}

@app.post("/log_sensor")
async def log_sensor(data: SensorData, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    new_data = HydroponicsData(
        timestamp=data.timestamp,
        water_level=data.water_level,
        tray_count=data.tray_count,
        humidity=data.humidity,
        temperature=data.temperature,
        ph=data.ph
    )
    db.add(new_data)
    db.commit()
    db.refresh(new_data)
    return {"message": "Data logged"}