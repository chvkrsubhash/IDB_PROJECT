import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from threading import Thread
import time
import pandas as pd

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define the CNN model
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Load pre-trained weights
emotion_model.load_weights('model.h5')

# Set OpenCL usage to false
cv2.ocl.setUseOpenCL(False)

# Emotion and music dictionaries
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
music_dist = {
    0: "video/angry.csv",
    1: "video/disgusted.csv",
    2: "video/fearful.csv",
    3: "video/happy.csv",
    4: "video/neutral.csv",
    5: "video/sad.csv",
    6: "video/surprised.csv"
}

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

class VideoCamera:
    def __init__(self):
        self.last_recognition_time = 0
        self.last_video_update_time = 0
        self.interval = 0 
        self.current_emotion_index = 4

    def get_frame(self):
        global df1
        cap1 = WebcamVideoStream(src=0).start()
        
        while True:
            image = cap1.read()
            image = cv2.resize(image, (600, 500))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

            current_time = time.time()

            if current_time - self.last_recognition_time >= self.interval:
                for (x, y, w, h) in face_rects:
                    cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
                    roi_gray_frame = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                    prediction = emotion_model.predict(cropped_img)

                    maxindex = int(np.argmax(prediction))
                    self.current_emotion_index = maxindex
                    cv2.putText(image, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                self.last_recognition_time = current_time

            if current_time - self.last_video_update_time >= self.interval:
                df1 = music_rec(self.current_emotion_index)
                self.last_video_update_time = current_time

            last_frame1 = image.copy()
            img = Image.fromarray(last_frame1)
            img = np.array(img)
            ret, jpeg = cv2.imencode('.jpg', img)
            
            time.sleep(1)  
            
            return jpeg.tobytes(), df1

def music_rec(emotion_index):
    df = pd.read_csv(music_dist[emotion_index])
    df = df[['Artist']]
    df = df.head(1)
    return df
