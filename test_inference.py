import torch
import torch._utils
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import os

# --- HEIC SUPPORT ---
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    print("HEIC support loaded.")
except ImportError:
    print("pillow-heif not installed. If you have .heic files, run: pip install pillow-heif")

# ==========================================
# 1. THE CNN ARCHITECTURE (With BatchNorm)
# ==========================================
class FaceEmbeddingCNN(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceEmbeddingCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 14 * 14, embedding_size)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

# ==========================================
# 2. SETUP & LOAD WEIGHTS
# ==========================================
device = torch.device('cpu') 

model_path = 'arcface_public_trained_cnn.pth'
model = FaceEmbeddingCNN().to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Successfully loaded weights from {model_path}")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find {model_path}. Make sure it is in the same folder as this script.")
    exit()
except RuntimeError as e:
    print(f"❌ ERROR: Architecture mismatch. Did you download the newest .pth file? Details: {e}")
    exit()

# Setup MTCNN for face cropping (resizes to 112x112 automatically)
mtcnn = MTCNN(image_size=112, margin=0, keep_all=False, device=device)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_embedding(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"  [Error] Could not open {image_path}: {e}")
        return None
        
    face_tensor = mtcnn(img) 
    if face_tensor is None:
        return None
        
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding.numpy().flatten()

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# ==========================================
# 4. ENROLLMENT & AUTO-HOLD-OUT
# ==========================================
print("\n--- Starting Enrollment (Holding out 1 image per person) ---")
enrolled_db = {}
held_out_tests = {} 

dataset_path = "TRAINING PICS" 

if not os.path.exists(dataset_path):
    print(f"❌ ERROR: Could not find the folder '{dataset_path}'.")
    exit()

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder) or person_name.startswith('.'):
        continue
        
    all_images = [f for f in os.listdir(person_folder) if not f.startswith('.')]
    
    if len(all_images) < 2:
        print(f"Skipping '{person_name}': Need at least 2 images.")
        continue

    test_img_file = all_images.pop() 
    held_out_tests[person_name] = os.path.join(person_folder, test_img_file)
    
    embeddings = []
    valid_files = 0
    
    for img_file in all_images:
        img_path = os.path.join(person_folder, img_file)
        emb = get_embedding(img_path)
        
        if emb is not None:
            embeddings.append(emb)
            valid_files += 1
            
    if embeddings:
        enrolled_db[person_name] = np.mean(embeddings, axis=0)
        print(f"✅ Enrolled '{person_name}' using {valid_files} images. (Held out 1 for testing).")

# ==========================================
# 5. INFERENCE FUNCTION
# ==========================================
def recognize_face(test_image_path, threshold=0.65):
    test_emb = get_embedding(test_image_path)
    
    if test_emb is None:
        return "No face detected", -1.0
        
    best_match = "Unknown"
    highest_sim = -1.0
    
    for name, enrolled_emb in enrolled_db.items():
        sim = cosine_similarity(test_emb, enrolled_emb)
        if sim > highest_sim:
            highest_sim = sim
            best_match = name
            
    if highest_sim < threshold:
        return "Unknown", highest_sim
    
    return best_match, highest_sim

# ==========================================
# 6. AUTOMATED TESTING ON HELD-OUT IMAGES
# ==========================================
print("\n" + "="*70)
print("--- AUTOMATED ATTENDANCE TEST (Held-Out Images) ---")
print("="*70)

correct_predictions = 0
total_tests = len(held_out_tests)

for true_name, test_path in held_out_tests.items():
    predicted_name, score = recognize_face(test_path)
    
    filename = os.path.basename(test_path)
    
    if predicted_name == true_name:
        status = "✅ MATCH"
        correct_predictions += 1
    elif predicted_name == "Unknown":
        status = "❌ REJECTED (Threshold too high)"
    else:
        status = f"❌ WRONG (Matched to {predicted_name})"
        
    print(f"True: {true_name:<15} | Test Img: {filename:<15} | Predicted: {predicted_name:<15} | Score: {score:.3f} | {status}")

print("\n" + "="*70)
accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
print(f"FINAL ACCURACY: {accuracy:.2f}% ({correct_predictions}/{total_tests})")
print("="*70)
