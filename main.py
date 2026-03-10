import subprocess
import json
import pyaudio
import time
import serial
import threading
from vosk import Model, KaldiRecognizer
from datetime import datetime
import cv2
from ultralytics import YOLO

PERSON_ID = 0

yolo_model = YOLO("yolov8n_ncnn_model")

person_width = 30 # cm
focal_length = 840

align_tolerance = 50
follow_distance = 100 # cm

cap = cv2.VideoCapture(0)
cap.set(3, 640) # W
cap.set(4, 480) # H

ser = serial.Serial('/dev/ttyACM0', 9600) # Arduino serial port
MIC_INDEX = 1
RATE = 16000
WAKE_WORD = "hey agent"
ACTIVE_TIMEOUT = 10 # sec
VOSK_MODEL_PATH = "/home/jetson/Desktop/vosk-model-small-en-us-0.15"

vosk_model = Model(VOSK_MODEL_PATH)
rec = KaldiRecognizer(vosk_model, RATE)

# Microphone
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=8000,
                input_device_index=MIC_INDEX)
stream.start_stream()

# Speaking
active_listen_mode = False
last_active_time = 0
# Following
follow_mode = False

person_detected = False
center_x = 0
distance = 0

def speak(text):
    subprocess.run(["espeak", "-s200", "-a200", text])

'''
def handle_command(text):
    if "what day is it" in text:
        current_day = datetime.now().strftime("%A")
        speak(f"Today is {current_day}")

    elif "what's the date" in text:
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        speak(f"Today is {current_date}")
    
    elif "what's your favorite food" in text:
        speak("My favorite food is pizza! I love carbs.")

    elif "who created you" in text:
        speak("I was created by a developer named Jonathan.")

    elif "who is casper" in text:
        speak("Casper is Jonathan's favorite dog. He's a really cute white fluffball.")'''

def vision_loop():
    global person_detected, center_x, distance, follow_mode
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(frame, imgsz=320, verbose=False)[0]
        person = None

        # Use bb for coords
        for result in results.boxes:
            if int(result.cls) == PERSON_ID:
                person = result
                break
        if person is not None:
            x1, y1, x2, y2 = map(int, person.xyxy[0])
            center_x = (x1 + x2) // 2
            width = x2 - x1
            distance = (person_width * focal_length) / width
            person_detected = True
        else:
            person_detected = False

        if follow_mode and person_detected:
            if center_x < 320 - align_tolerance:
                print("Turn left")
                ser.write(b'L')
            elif center_x > 320 + align_tolerance:
                print("Turn right")
                ser.write(b'R')
            else:
                print("Aligned")
                ser.write(b'S')
            if distance > follow_distance:
                print("Go forward")
                ser.write(b'F')
            else:
                print("Stop")
                ser.write(b'S')

# Start vision thread
vision_thread = threading.Thread(target=vision_loop)
vision_thread.daemon = True
vision_thread.start()

while True:
    data = stream.read(4000, exception_on_overflow=False)

     # Timeout control
    if active_listen_mode and (time.time() - last_active_time > ACTIVE_TIMEOUT):
        active_listen_mode = False
        print("Going back to sleep...")

    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        text = result.get("text", "").lower()

        if not text:
            continue

        print(f"Recognized: {text}")

        # Wake word detection
        if not active_listen_mode:
            if WAKE_WORD in text:
                active_listen_mode = True
                last_active_time = time.time()
                print("Activated!")
                speak("Yes?")
        
        # Active mode
        else:
            if "follow me" in text:
                speak("I will follow you!")
                follow_mode = True
            elif "stop" in text:
                speak("Stopping all actions.")
                ser.write(b'S')
                follow_mode = False
            elif "what day is it" in text:
                current_day = datetime.now().strftime("%A")
                speak(f"Today is {current_day}")
            elif "what's the date" in text:
                current_date = datetime.now().strftime("%A, %B %d, %Y")
                speak(f"Today is {current_date}")
            elif "what's your favorite food" in text:
                speak("My favorite food is pizza! I love carbs.")
            elif "who created you" in text:
                speak("I was created by a developer named Jonathan.")
            elif "who is casper" in text:
                speak("Casper is Jonathan's favorite dog. He's a really cute white fluffball.")

            last_active_time = time.time()

# CTRL+C to exit since no break in loop rn
cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()