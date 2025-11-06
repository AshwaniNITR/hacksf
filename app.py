from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io

app = Flask(__name__)

# -------------------- Setup --------------------
device = torch.device("cpu")

# Load models
print("🚀 Loading models...")
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("✅ Models loaded successfully!")

# -------------------- Helper Functions --------------------
def extract_face_embedding(img_bytes):
    """Detects, crops, aligns, and extracts facial embedding."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    face = mtcnn(img)
    if face is None:
        return None

    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face)
    emb = F.normalize(emb, p=2, dim=1)
    return emb


def cosine_similarity(a, b):
    return F.cosine_similarity(a, b).item()


# -------------------- ROUTE 1: Compare two faces --------------------
@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({"error": "Please upload both image1 and image2"}), 400

        img1_bytes = request.files['image1'].read()
        img2_bytes = request.files['image2'].read()

        emb1 = extract_face_embedding(img1_bytes)
        emb2 = extract_face_embedding(img2_bytes)

        if emb1 is None or emb2 is None:
            return jsonify({"error": "Face not detected in one or both images"}), 400

        sim = cosine_similarity(emb1, emb2)
        threshold = 0.60
        result = "Same person" if sim > threshold else "Different person"

        return jsonify({
            "similarity": round(sim, 4),
            "threshold": threshold,
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------- ROUTE 2: Get embedding of a single image --------------------
@app.route('/get_embeddings', methods=['POST'])
def get_embeddings():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Please upload an image with key 'image'"}), 400

        img_bytes = request.files['image'].read()
        emb = extract_face_embedding(img_bytes)

        if emb is None:
            return jsonify({"error": "No face detected in the image"}), 400

        # Convert tensor to Python list for JSON serialization
        embedding_list = emb.squeeze(0).tolist()

        return jsonify({
            "embedding_dim": len(embedding_list),
            "embedding_vector": embedding_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------- Run --------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
