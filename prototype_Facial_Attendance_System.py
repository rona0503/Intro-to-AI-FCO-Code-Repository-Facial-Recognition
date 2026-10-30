import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import os
import time
from datetime import datetime
import csv

# --- KEYBOARD LISTENER SETUP ---
try:
    import keyboard
    print("✅ Global keyboard listener loaded.")
except ImportError:
    print("❌ ERROR: The 'keyboard' library is missing. Run: pip install keyboard")
    exit()

# This variable will store the key press reliably
key_action = None

def handle_key(key):
    global key_action
    key_action = key

# Register hotkeys (these trigger instantly on key down)
keyboard.add_hotkey('q', lambda: handle_key('q'))
keyboard.add_hotkey('e', lambda: handle_key('e'))
keyboard.add_hotkey('s', lambda: handle_key('s'))

# --- HEIC IMAGE SUPPORT ---
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    print("✅ HEIC support loaded.")
except ImportError:
    print("⚠️ pillow-heif not installed. HEIC images may not load. Run: pip install pillow-heif")

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
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
# 2. INITIALIZATION AND MODEL LOADING
# ==========================================
print("\n--- Initializing System ---")
device = torch.device('cpu') 
model_path = 'arcface_public_trained_cnn.pth'

print(f"Loading model weights from '{model_path}'...")
model = FaceEmbeddingCNN().to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find '{model_path}'. Ensure it is in the same directory.")
    exit()

print("Initializing MTCNN face detector...")
mtcnn = MTCNN(image_size=112, margin=0, keep_all=False, device=device)
print("✅ MTCNN initialized.")

# ==========================================
# 3. DATABASE LOADING & HELPER FUNCTIONS
# ==========================================
enrolled_db = {}
image_counts = {} 
dataset_path = "TRAINING PICS"

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def get_embedding_from_pil(img_pil):
    face_tensor = mtcnn(img_pil) 
    if face_tensor is None: return None
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding.numpy().flatten()

print(f"\nScanning '{dataset_path}' for existing enrollments...")
if os.path.exists(dataset_path):
    count = 0
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_folder) or person_name.startswith('.'): 
            continue
        
        embeddings = []
        for img_file in os.listdir(person_folder):
            if img_file.startswith('.'): 
                continue
            img = Image.open(os.path.join(person_folder, img_file)).convert('RGB')
            emb = get_embedding_from_pil(img)
            if emb is not None: 
                embeddings.append(emb)
            
        if embeddings:
            enrolled_db[person_name] = np.mean(embeddings, axis=0)
            image_counts[person_name] = len(embeddings)
            print(f"  -> Loaded: {person_name} ({len(embeddings)} images)")
            count += 1
    
    print(f"✅ Database loaded. Total identities: {count}")
else:
    print(f"⚠️ Directory '{dataset_path}' not found. Starting with an empty database.")

# ==========================================
# 4. CAMERA AND ATTENDANCE LOGIC
# ==========================================
threshold = 0.65
log_file = "attendance.csv"
last_seen = {}
last_logged = {}

print("\n--- Starting Camera ---")
print("Controls (Global Hotkeys, no need to click the window):")
print("  'q' - Quit")
print("  'e' - Enroll an UNKNOWN face")
print("  's' - Save a snapshot of a RECOGNIZED face")

window_name = "Face Attendance System"
cv2.namedWindow(window_name)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Failed to grab frame from camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    boxes, _ = mtcnn.detect(pil_img)
    
    status = "No Face"
    top_guess_name = None
    highest_sim = -1.0
    test_emb = None
    x1, y1, x2, y2 = 0, 0, 0, 0

    if boxes is not None:
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]

        test_emb = get_embedding_from_pil(pil_img)

        if test_emb is not None:
            for name, enrolled_emb in enrolled_db.items():
                sim = cosine_similarity(test_emb, enrolled_emb)
                if sim > highest_sim:
                    highest_sim = sim
                    top_guess_name = name

            if highest_sim >= threshold:
                status = "Known"
                color = (0, 255, 0) 
                display_text = f"{top_guess_name} ({highest_sim:.2f})"
                
                current_time = time.time()
                previous_last_seen = last_seen.get(top_guess_name, 0)
                last_seen[top_guess_name] = current_time

                if top_guess_name not in last_logged or (current_time - previous_last_seen > 30):
                    last_logged[top_guess_name] = current_time
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file_exists = os.path.isfile(log_file)
                    
                    with open(log_file, "a", newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["Timestamp", "Name"])
                        writer.writerow([timestamp, top_guess_name])
                        
                    print(f"✅ Logged: {top_guess_name} at {timestamp} to CSV.")
                
                cv2.putText(frame, "Press 's' to save snapshot", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            else:
                status = "Unknown"
                color = (0, 0, 255) 
                display_text = f"Unknown ({highest_sim:.2f})"
                cv2.putText(frame, "Press 'e' to enroll", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow(window_name, frame)
    
    # Check if a hotkey was triggered
    if key_action is not None:
        action = key_action
        key_action = None  # Reset the action immediately
        
        if action == 'q':
            break
            
        elif action == 's' and status == "Known":
            new_folder = os.path.join(dataset_path, top_guess_name)
            os.makedirs(new_folder, exist_ok=True)
            face_crop = pil_img.crop((x1, y1, x2, y2)).resize((112, 112))
            save_path = os.path.join(new_folder, f"snapshot_{int(time.time())}.jpg")
            face_crop.save(save_path)
            
            old_count = image_counts.get(top_guess_name, 1)
            new_avg = (enrolled_db[top_guess_name] * old_count + test_emb) / (old_count + 1)
            enrolled_db[top_guess_name] = new_avg / np.linalg.norm(new_avg)
            image_counts[top_guess_name] = old_count + 1
            
            print(f"📸 Snapshot saved for {top_guess_name}! Added to database ({save_path}).")

        elif action == 'e' and status == "Unknown":
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n--- Enrollment Initiated ---")
            if top_guess_name is not None:
                print(f"System Recommendation: This face looks closest to '{top_guess_name}' (Score: {highest_sim:.2f}).")
                print(f"If this is {top_guess_name}, type their name below to add this photo to their folder.")
            else:
                print("System Recommendation: No strong matches in the database.")
                
            new_name = input("Enter name for enrollment (or press Enter to cancel): ").strip()
            
            if new_name:
                new_folder = os.path.join(dataset_path, new_name)
                os.makedirs(new_folder, exist_ok=True)
                face_crop = pil_img.crop((x1, y1, x2, y2)).resize((112, 112))
                save_path = os.path.join(new_folder, f"enroll_{int(time.time())}.jpg")
                face_crop.save(save_path)
                
                enrolled_db[new_name] = test_emb
                image_counts[new_name] = 1
                print(f"✅ Enrolled {new_name}! Image saved to {save_path}")
            else:
                print("Enrollment cancelled.")
            
            print("Resuming camera...")
            cv2.namedWindow(window_name)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            cap = cv2.VideoCapture(0)
            
            # Clear any key strokes that happened while typing the name
            key_action = None

    # Allow OpenCV to render
    cv2.waitKey(1)

# Clean up
try:
    keyboard.remove_hotkey('q')
    keyboard.remove_hotkey('e')
    keyboard.remove_hotkey('s')
except:
    pass

cap.release()
cv2.destroyAllWindows()
print("\nCamera stopped. System shut down.")
