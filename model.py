import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
import os

class EmotionRecognition:
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.model = self.create_model()
    
    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        
        return model

    def prepare_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(detected_faces) > 0:
            (x, y, w, h) = detected_faces[0]
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            
            return roi_gray, (x, y, w, h)
        
        return None, None

    def identify_emotion(self, image):
        processed_image, face_coordinates = self.prepare_image(image)
        
        if processed_image is not None:
            predictions = self.model.predict(processed_image)[0]
            emotion_index = np.argmax(predictions)
            detected_emotion = self.emotion_labels[emotion_index]
            
            return detected_emotion, face_coordinates, predictions[emotion_index]
        
        return None, None, None

    def run_video_capture(self):
        video_capture = cv2.VideoCapture(0)
        
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            
            emotion, face_coordinates, confidence = self.identify_emotion(frame)
            
            if emotion and face_coordinates:
                (x, y, w, h) = face_coordinates
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text_display = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, text_display, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
            cv2.imshow('Emotion Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    emotion_recognizer = EmotionRecognition()
    emotion_recognizer.run_video_capture()
