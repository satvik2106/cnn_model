import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import cv2
import base64
import os
from pymongo import MongoClient
import gcsfs
import tempfile

# MongoDB connection
mongo_client = MongoClient("mongodb+srv://satvikvattipalli1311:8I4SOudfJO8n8fIp@signare.w1j4f.mongodb.net/?retryWrites=true&w=majority&appName=Signare")
db = mongo_client.test
collection = db.accounts

# Google Cloud Storage configuration
bucket_name = "signature_verification_storage"
model_file_name = "Signature_verification(DL model).h5"

# Access model from Google Cloud Storage
fs = gcsfs.GCSFileSystem()
model_path = f"gs://{bucket_name}/{model_file_name}"

try:
    # Create a temporary file to store the model locally
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        print("Downloading model from Google Cloud Storage...")
        
        # Download the model from GCS
        with fs.open(model_path, 'rb') as gcs_file:
            temp_file.write(gcs_file.read())
        temp_file_name = temp_file.name

    print("Model downloaded successfully. Loading the model...")

    # Load the model from the temporary file
    trained_model = load_model(temp_file_name)
    print("Model loaded successfully.")

finally:
    # Clean up the temporary file
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)
        print("Temporary file deleted.")

# Create Flask app
app = Flask(__name__)
CORS(app)

# Health check endpoint to verify DL model functionality
@app.route('/api/health-check', methods=['GET'])
def health_check():
    try:
        # Dummy input to check if the model works
        dummy_input = np.zeros((1, 224, 224, 3))  # Replace dimensions with your model's expected input
        dummy_output = trained_model.predict(dummy_input)
        
        # Return a success message
        return jsonify({"message": "DL model is working.", "dummy_output_shape": dummy_output.shape}), 200
    except Exception as e:
        return jsonify({"error": f"Model check failed: {str(e)}"}), 500

# Feature extraction model
def create_advanced_embedding_model(trained_model):
    feature_extractor = Model(inputs=trained_model.inputs, outputs=trained_model.layers[-8].output)
    return feature_extractor

feature_extractor = create_advanced_embedding_model(trained_model)

# Image preprocessing function
def preprocess_image(image_data):
    img_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Signature verification endpoint
@app.route('/api/signature/verify', methods=['POST'])
def verify_signature():
    try:
        account_number = request.form['account_number']
        verifying_signature_file = request.files['verifying_signature']

        # Retrieve stored signature from MongoDB
        record = collection.find_one({"accountNumber": str(account_number)})
        if not record or "image" not in record:
            raise ValueError("Signature not found in the database.")

        # Decode stored signature
        stored_signature_data = base64.b64decode(record["image"])

        # Preprocess images
        stored_image = preprocess_image(stored_signature_data)
        verifying_image = preprocess_image(verifying_signature_file.read())

        # Extract embeddings
        stored_embedding = feature_extractor.predict(stored_image).flatten()
        verifying_embedding = feature_extractor.predict(verifying_image).flatten()

        # Calculate cosine similarity
        similarity = cosine_similarity([stored_embedding], [verifying_embedding])[0][0]

        # Threshold for decision
        threshold = 0.8
        result = "Genuine" if similarity > threshold else "Forged"

        return jsonify({"similarity": float(similarity), "result": result})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)  # Set debug=True only for development
