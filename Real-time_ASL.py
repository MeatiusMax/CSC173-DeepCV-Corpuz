import cv2
import numpy as np
from tensorflow.keras.models import load_model

#configurations
MODEL_PATH = "model/asl_cnn_model.h5"
IMG_SIZE = 64 
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing'
]

# load the trained model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

#webcam setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # mirror effect
    frame = cv2.flip(frame, 1)

    # ROI for hand gesture
    x_start, y_start, width, height = 100, 100, 300, 300
    roi = frame[y_start:y_start+height, x_start:x_start+width]

    # Prepare the ROI for prediction
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_normalized = roi_resized.astype('float32') / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)  

    #prediction
    pred = model.predict(roi_expanded)
    class_idx = np.argmax(pred)
    predicted_letter = CLASS_NAMES[class_idx]
    confidence = np.max(pred)

    # display the prediction
    cv2.rectangle(frame, (x_start, y_start), (x_start+width, y_start+height), (0, 255, 0), 2)
    cv2.putText(frame, f"{predicted_letter} ({confidence*100:.1f}%)", 
                (x_start, y_start-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Real-Time Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
