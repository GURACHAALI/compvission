import cv2 as cv
import numpy as np
import tensorflow as tf

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

model_path = 'model.hdf5' 
 
emotion_model = tf.keras.models.load_model(model_path, compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_and_classify_emotions(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        
        resized_face = cv.resize(face_roi, (64, 64))
        
        normalized_face = resized_face / 255.0
        
        reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))
    
        emotion_scores = emotion_model.predict(reshaped_face)
        predicted_emotion = emotion_labels[np.argmax(emotion_scores)]
        
        cv.rectangle(image, (x, y), (x+w, y+h), ( 0,255, 0), 2)
        cv.putText(image, predicted_emotion, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv.LINE_AA)
    
    return image

# Example usage:
if __name__ == "__main__":
    image_path = 'kido.jpg'
    image = cv.imread(image_path)
    
    if image is None:
        print(f'Failed to load image at {image_path}')
    else:
        image_with_emotions = detect_and_classify_emotions(image)
        
        # Display the image with detected faces and classified emotions
        cv.imshow('Describe Emotions', image_with_emotions)
        cv.waitKey(0)
        cv.destroyAllWindows()
