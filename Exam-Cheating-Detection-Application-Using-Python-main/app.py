import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import time
import mediapipe as mp
from pathlib import Path
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize detection models
yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8n.pt')
except ImportError:
    print("Warning: YOLO not available")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# MediaPipe initialization
mp_face_mesh = None
mp_hands = None

try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options
    
    mediapipe_models_dir = Path.home() / '.mediapipe'
    face_model_path = mediapipe_models_dir / 'face_landmarker.task'
    hand_model_path = mediapipe_models_dir / 'hand_landmarker.task'
    
    if face_model_path.exists() and hand_model_path.exists():
        base_opts = base_options.BaseOptions(model_asset_path=str(face_model_path))
        face_options = vision.FaceLandmarkerOptions(base_options=base_opts, num_faces=2)
        mp_face_mesh = vision.FaceLandmarker.create_from_options(face_options)
        
        base_opts_hand = base_options.BaseOptions(model_asset_path=str(hand_model_path))
        hand_options = vision.HandLandmarkerOptions(base_options=base_opts_hand, num_hands=2)
        mp_hands = vision.HandLandmarker.create_from_options(hand_options)
except Exception as e:
    print(f"MediaPipe initialization warning: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """Process uploaded video file for cheating detection"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return jsonify({'error': 'Invalid video format'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process video
        results = process_video_file(filepath)
        
        # Cleanup
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video_file(filepath):
    """Analyze video file for cheating behavior"""
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        return {'error': 'Could not open video file'}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cheating_frames = 0
    looking_away_count = 0
    multiple_faces_count = 0
    phone_detected_count = 0
    
    detections = {
        'total_frames': total_frames,
        'fps': fps,
        'duration_seconds': total_frames / fps if fps > 0 else 0,
        'cheating_events': [],
        'summary': {
            'cheating_frames': 0,
            'looking_away_count': 0,
            'multiple_faces_count': 0,
            'phone_detected_count': 0,
            'cheating_percentage': 0.0
        }
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps if fps > 0 else 0
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        frame_event = {
            'frame': frame_count,
            'timestamp': timestamp,
            'violations': []
        }
        
        # Check for multiple faces
        if len(faces) > 1:
            multiple_faces_count += 1
            frame_event['violations'].append('MULTIPLE_FACES')
            cheating_frames += 1
        elif len(faces) == 1:
            face_x, face_y, face_w, face_h = faces[0]
            
            # Check for phone/objects using YOLO
            if yolo_model:
                try:
                    results = yolo_model(frame, conf=0.5)
                    for result in results:
                        for box in result.boxes:
                            class_name = result.names[int(box.cls)]
                            if class_name in ['cell phone', 'phone', 'book']:
                                phone_detected_count += 1
                                frame_event['violations'].append(f'OBJECT_DETECTED:{class_name}')
                                cheating_frames += 1
                except:
                    pass
        
        if frame_event['violations']:
            detections['cheating_events'].append(frame_event)
    
    cap.release()
    
    # Calculate summary
    if total_frames > 0:
        cheating_percentage = (cheating_frames / total_frames) * 100
    else:
        cheating_percentage = 0
    
    detections['summary'] = {
        'cheating_frames': cheating_frames,
        'looking_away_count': looking_away_count,
        'multiple_faces_count': multiple_faces_count,
        'phone_detected_count': phone_detected_count,
        'cheating_percentage': round(cheating_percentage, 2),
        'status': 'SUSPICIOUS' if cheating_percentage > 10 else 'NORMAL'
    }
    
    return detections

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
