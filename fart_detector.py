#!/usr/bin/env python3
"""
Simple Raspberry Pi Fart Detector using YAMNet
Continuously listens to USB microphone and detects fart sounds
"""

import sounddevice as sd
import numpy as np
import tensorflow as tf
import csv
import time

# Global configuration constants
SAMPLE_RATE = 16000  # Hz - YAMNet expects 16kHz
DURATION = 2.0  # seconds - longer window for better detection
CHANNELS = 1
DEVICE_ID = None  # None for default device
YAMNET_MODEL_PATH = 'yamnet_model'  # Local YAMNet model path
CLASS_MAP_PATH = 'yamnet_model/yamnet_class_map.csv'
DETECTION_THRESHOLD = 0.3  # Threshold for fart detection
BLOCK_SIZE = int(SAMPLE_RATE * DURATION)

def load_class_map():
    """Load YAMNet class names from CSV file"""
    try:
        class_map = {}
        with open(CLASS_MAP_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_map[int(row['index'])] = row['display_name']
        
        # Find fart class indices
        fart_indices = [i for i, name in class_map.items() if 'fart' in name.lower()]
        print(f"Fart class indices: {fart_indices}")
        return class_map, fart_indices
    except Exception as e:
        print(f"Failed to load class map: {e}")
        return {}, []

def load_yamnet_model():
    """Load the YAMNet model"""
    try:
        model = tf.saved_model.load(YAMNET_MODEL_PATH)
        print(f"YAMNet model loaded successfully from {YAMNET_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Failed to load YAMNet model: {e}")
        return None

def preprocess_audio(audio):
    """Preprocess audio for YAMNet model"""
    # Ensure audio is mono and float32
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    
    # Normalize audio (like the example)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio

def predict_fart(yamnet_model, audio, fart_indices):
    """Predict if audio contains a fart using YAMNet"""
    if yamnet_model is None:
        # Return random prediction for demonstration
        return np.random.random()
    
    try:
        # Run YAMNet inference
        scores, embeddings, spectrogram = yamnet_model(audio)
        scores_np = scores.numpy()
        
        # Average scores across frames
        mean_scores = np.mean(scores_np, axis=0)
        
        # Check fart probability
        max_fart_score = 0.0
        for idx in fart_indices:
            fart_score = mean_scores[idx]
            max_fart_score = max(max_fart_score, fart_score)
        
        return max_fart_score
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return 0.0

def main():
    """Main detection loop using YAMNet"""
    print("=" * 50)
    print("Simple Raspberry Pi Fart Detector using YAMNet")
    print("=" * 50)
    
    # List available audio devices
    try:
        devices = sd.query_devices()
        print(f"Available audio devices: {len(devices)}")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  Device {i}: {device['name']} (inputs: {device['max_input_channels']})")
    except Exception as e:
        print(f"Failed to query audio devices: {e}")
    
    # Load class map and find fart indices
    class_map, fart_indices = load_class_map()
    
    # Load YAMNet model
    yamnet_model = load_yamnet_model()
    
    if yamnet_model is None:
        print("Failed to load YAMNet model. Exiting.")
        return
    
    print(f"Starting detection loop...")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Threshold: {DETECTION_THRESHOLD}")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Use continuous stream
        with sd.InputStream(channels=CHANNELS, 
                          samplerate=SAMPLE_RATE, 
                          blocksize=BLOCK_SIZE, 
                          dtype='float32',
                          device=DEVICE_ID) as stream:
            print("Listening for farts...")
            
            while True:
                # Read audio block from stream
                audio, overflowed = stream.read(BLOCK_SIZE)
                audio = audio.flatten()
                
                # Preprocess audio
                processed_audio = preprocess_audio(audio)
                
                # Make prediction using YAMNet
                confidence = predict_fart(yamnet_model, processed_audio, fart_indices)
                
                # Check if fart is detected
                if confidence >= DETECTION_THRESHOLD:
                    print(f"🚨 FART DETECTED! (confidence: {confidence:.3f})")
                else:
                    print(f"... quiet ({confidence:.3f})")
                
    except KeyboardInterrupt:
        print("\nStopping detection...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
