import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("Model_B\\model_B.keras")
# model = tf.keras.models.load_model("Model_B\\MobileNetV2_CMU_Face.keras")

CLASS_NAMES = ["straight", "left", "right", "up"]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Convert webcam image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to model input size: width=64, height=60
    image = cv2.resize(gray, (64, 60))

    # Normalize exactly like training data
    image = image.astype(np.float32) / 255.0

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Prediction
    prediction = model.predict(image, verbose=0)

    class_index = np.argmax(prediction[0])
    confidence = prediction[0][class_index]

    label = CLASS_NAMES[class_index]

    # Display result
    text = f"{label}: {confidence * 100:.1f}%"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Pose Classifier", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()