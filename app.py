import os, io, base64, logging
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs("uploads", exist_ok=True)

MODEL_PATH = "digit_model.h5"

def train_and_save():
    logging.info("Training model...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train, 10)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(28,28,1)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Conv2D(64, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=5, batch_size=128,
              validation_data=(X_val, y_val),
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
              verbose=1)
    model.save(MODEL_PATH)
    logging.info("Model trained and saved.")
    return model

# Load or train model on startup
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    logging.info("Model loaded from file.")
else:
    model = train_and_save()

def preprocess_image(image_data):
    pil_img = Image.open(io.BytesIO(image_data)).convert("RGBA")
    img_np  = np.array(pil_img)
    bg        = np.ones_like(img_np[:,:,:3], dtype=np.uint8) * 255
    alpha     = img_np[:,:,3:4] / 255.0
    composite = (img_np[:,:,:3] * alpha + bg * (1 - alpha)).astype(np.uint8)
    gray      = cv2.cvtColor(composite, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 2:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary  = np.where(labels == largest, 255, 0).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 6
        x1=max(x-pad,0); y1=max(y-pad,0)
        x2=min(x+w+pad, binary.shape[1]); y2=min(y+h+pad, binary.shape[0])
        binary = binary[y1:y2, x1:x2]
    h, w   = binary.shape
    side   = max(h, w) + 24
    square = np.zeros((side, side), dtype=np.uint8)
    square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = binary
    resized    = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    normalised = resized.astype("float32") / 255.0
    return normalised.reshape(1, 28, 28, 1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503
    try:
        if request.is_json:
            data    = request.get_json()
            img_b64 = data.get("image", "")
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(img_b64)
        elif "file" in request.files:
            image_bytes = request.files["file"].read()
        else:
            return jsonify({"error": "No image data received."}), 400
        img_array   = preprocess_image(image_bytes)
        predictions = model.predict(img_array, verbose=0)[0]
        digit       = int(np.argmax(predictions))
        confidence  = float(predictions[digit])
        all_probs   = {str(i): round(float(predictions[i])*100, 2) for i in range(10)}
        return jsonify({
            "prediction": digit,
            "confidence": f"{confidence*100:.2f}%",
            "all_probabilities": all_probs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)