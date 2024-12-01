import streamlit as st
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from PIL import Image

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
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Detect faces with improved parameters
        detected_faces = self.face_detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(60, 60),
            maxSize=(800, 800),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(detected_faces) > 0:
            # Sort faces by size and take the largest
            detected_faces = sorted(detected_faces, key=lambda x: x[2]*x[3], reverse=True)
            (x, y, w, h) = detected_faces[0]
            
            # Add padding
            padding = int(0.1 * w)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            # Extract and process face ROI
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = cv2.equalizeHist(roi_gray)
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            
            return roi_gray, (x, y, w, h)
        
        return None, None

    def identify_emotion(self, image):
        processed_image, face_coordinates = self.prepare_image(image)
        
        if processed_image is not None:
            predictions = self.model.predict(processed_image, verbose=0)[0]
            emotion_index = np.argmax(predictions)
            detected_emotion = self.emotion_labels[emotion_index]
            
            return detected_emotion, face_coordinates, predictions[emotion_index]
        
        return None, None, None

def init_session_state():
    # Initialize session state variables
    defaults = {
        'emotion_stats': {},
        'total_detections': 0,
        'camera_on': False,
        'emotion_history': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def plot_emotion_trends(emotion_history):
    """Create an emotion trend visualization."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    emotions = [entry['emotion'] for entry in emotion_history]
    plt.title('Emotion Detection Trends')
    plt.xlabel('Detection Sequence')
    plt.ylabel('Detected Emotion')
    plt.yticks(range(len(set(emotions))), list(set(emotions)))
    plt.plot(range(len(emotions)), [list(set(emotions)).index(e) for e in emotions], marker='o')
    plt.tight_layout()
    return plt

def main():
    st.set_page_config(page_title="Advanced Emotion Recognition", layout="wide")
    
    # Initialize session state
    init_session_state()
    
    st.title("Advanced Real-time Emotion Recognition")
    st.write("Detect and analyze emotions in real-time using your webcam.")
    
    # Load emotion recognizer
    @st.cache_resource
    def load_emotion_recognizer():
        return EmotionRecognition()
    
    emotion_recognizer = load_emotion_recognizer()
    
    # Sidebar controls
    st.sidebar.title("Controls")
    camera_toggle = st.sidebar.button("Toggle Camera")
    if camera_toggle:
        st.session_state.camera_on = not st.session_state.camera_on
    
    # Reset statistics and history
    if st.sidebar.button("Reset Data"):
        st.session_state.emotion_stats = {}
        st.session_state.total_detections = 0
        st.session_state.emotion_history = []
    
    # Camera feed
    if st.session_state.camera_on:
        camera_column, stats_column = st.columns([2, 1])
        
        with camera_column:
            camera_feed = st.camera_input("Camera Feed")
            
            if camera_feed is not None:
                # Convert PIL image to OpenCV
                image = np.array(Image.open(camera_feed))
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Detect emotion
                emotion, face_coordinates, confidence = emotion_recognizer.identify_emotion(frame)
                
                if emotion and face_coordinates:
                    (x, y, w, h) = face_coordinates
                    
                    # Update statistics and history
                    st.session_state.emotion_stats[emotion] = st.session_state.emotion_stats.get(emotion, 0) + 1
                    st.session_state.total_detections += 1
                    
                    # Track emotion history
                    emotion_entry = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'timestamp': len(st.session_state.emotion_history)
                    }
                    st.session_state.emotion_history.append(emotion_entry)
                    
                    # Draw rectangle and emotion label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text_display = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, text_display, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Convert back to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB", use_column_width=True)
        
        with stats_column:
            # Emotion Distribution
            if st.session_state.total_detections > 0:
                st.subheader("Emotion Statistics")
                for emotion, count in st.session_state.emotion_stats.items():
                    percentage = (count / st.session_state.total_detections) * 100
                    st.metric(
                        label=emotion,
                        value=f"{percentage:.1f}%",
                        delta=f"{count} detections"
                    )
            
            # Emotion Trend Plot
            if len(st.session_state.emotion_history) > 1:
                st.subheader("Emotion Trend")
                trend_plot = plot_emotion_trends(st.session_state.emotion_history)
                st.pyplot(trend_plot)

if __name__ == "__main__":
    main()