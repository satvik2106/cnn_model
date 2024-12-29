import os
import base64
import tempfile
import logging
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import gcsfs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection setup
MONGO_URI = "mongodb+srv://satvikvattipalli1311:8I4SOudfJO8n8fIp@signare.w1j4f.mongodb.net/?retryWrites=true&w=majority&appName=Signare"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["test"]
collection = db["accounts"]

# Google Cloud Storage configuration
BUCKET_NAME = "signature_verification_storage"
MODEL_FILE_NAME = "Signature_verification(DL model).h5"

# Access model from Google Cloud Storage
fs = gcsfs.GCSFileSystem()
model_path = f"gs://{BUCKET_NAME}/{MODEL_FILE_NAME}"

# Load model from Google Cloud Storage
def load_trained_model():
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
            with fs.open(model_path, 'rb') as gcs_file:
                temp_file.write(gcs_file.read())
            temp_file.close()
            model = load_model(temp_file.name)
            os.unlink(temp_file.name)  # Clean up after loading
        logger.info("Model successfully loaded from Google Cloud Storage.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

trained_model = load_trained_model()

# Create Flask app
app = Flask(__name__)
CORS(app)

# Feature extraction model setup
def create_embedding_model(trained_model):
    return Model(inputs=trained_model.inputs, outputs=trained_model.layers[-8].output)

feature_extractor = create_embedding_model(trained_model)

# Image preprocessing function
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Retrieve and store the signature from the database
def retrieve_signature(account_number):
    try:
        record = collection.find_one({"accountNumber": str(account_number)})
        if not record or "image" not in record:
            raise ValueError("Signature not found for the given account number.")
        signature_data = base64.b64decode(record["image"]) if isinstance(record["image"], str) else record["image"]
        signature_image = np.frombuffer(signature_data, dtype=np.uint8)
        signature_image = cv2.imdecode(signature_image, cv2.IMREAD_GRAYSCALE)
        signature_image = cv2.resize(signature_image, (256, 256))
        return signature_image
    except Exception as e:
        logger.error(f"Error retrieving signature: {e}")
        raise

# Signature verification endpoint
@app.route('/api/signature/verify', methods=['POST'])
def verify_signature():
    try:
        # Input validation
        if 'account_number' not in request.form or 'verifying_signature' not in request.files:
            return jsonify({"error": "Missing required parameters."}), 400

        account_number = request.form['account_number']
        verifying_signature_file = request.files['verifying_signature']

        # Retrieve stored signature
        stored_signature_image = retrieve_signature(account_number)

        # Save and preprocess the verifying signature
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            verifying_signature_file.save(temp_file.name)
            verifying_signature_image = preprocess_image(temp_file.name)
            os.unlink(temp_file.name)

        # Preprocess the stored signature
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            cv2.imwrite(temp_file.name, stored_signature_image)
            stored_signature_image = preprocess_image(temp_file.name)
            os.unlink(temp_file.name)

        # Extract embeddings and calculate similarity
        stored_embedding = feature_extractor.predict(stored_signature_image).flatten()
        verifying_embedding = feature_extractor.predict(verifying_signature_image).flatten()
        similarity = cosine_similarity([stored_embedding], [verifying_embedding])[0][0]

        # Determine result
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.8))
        result = "Genuine" if similarity > threshold else "Forged"

        return jsonify({"similarity": float(similarity), "result": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

# Health check endpoint
@app.route('/api/health-check', methods=['GET'])
def health_check():
    try:
        dummy_input = np.zeros((1, 224, 224, 3))
        trained_model.predict(dummy_input)
        return jsonify({"message": "DL model is working."}), 200
    except Exception as e:
        return jsonify({"error": f"Model check failed: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
