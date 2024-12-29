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
    # Use the system's temporary directory for storing the file
    temp_dir = tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(suffix=".h5", dir=temp_dir, delete=False) as temp_file:
        with fs.open(model_path, 'rb') as gcs_file:
            temp_file.write(gcs_file.read())
        temp_file.close()  # Explicitly close the file to ensure it's flushed
        trained_model = load_model(temp_file.name)
        os.unlink(temp_file.name)  # Remove the temporary file after loading the model
    print("Model loaded successfully from Google Cloud Storage.")
except Exception as e:
    raise Exception(f"Error loading the model from Google Cloud Storage: {e}")

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

# Function to retrieve and store the signature in a file
def retrieve_and_store_signature(account_number):
    record = collection.find_one({"accountNumber": str(account_number)})
    if record and "image" in record:
        signature_data = record["image"]
        if isinstance(signature_data, str):  # Decode base64 string
            signature_data = base64.b64decode(signature_data)
        stored_signature = np.frombuffer(signature_data, dtype=np.uint8)
        stored_signature = cv2.imdecode(stored_signature, cv2.IMREAD_GRAYSCALE)
        stored_signature = cv2.resize(stored_signature, (256, 256))  # Resize for consistency

        # Save the stored signature to a file
        stored_signature_path = "stored_signature.jpg"
        cv2.imwrite(stored_signature_path, stored_signature)

        return stored_signature_path
    else:
        raise ValueError("Signature not found in the database.")

# Feature extraction model
def create_advanced_embedding_model(trained_model):
    feature_extractor = Model(inputs=trained_model.inputs, outputs=trained_model.layers[-8].output)
    return feature_extractor

feature_extractor = create_advanced_embedding_model(trained_model)

# Image preprocessing function
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Signature verification endpoint
@app.route('/api/signature/verify', methods=['POST'])
def verify_signature():
    try:
        account_number = request.form['account_number']
        verifying_signature_file = request.files['verifying_signature']

        # Retrieve and save the stored signature
        stored_signature_path = retrieve_and_store_signature(account_number)

        # Save the verifying signature temporarily
        verifying_signature_path = "verifying_signature.jpg"
        verifying_signature_file.save(verifying_signature_path)

        # Preprocess images
        stored_image = preprocess_image(stored_signature_path)
        verifying_image = preprocess_image(verifying_signature_path)

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
