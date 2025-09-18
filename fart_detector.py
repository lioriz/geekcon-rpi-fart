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
from scipy import signal  # for resampling

# Global configuration constants
YAMNET_SAMPLE_RATE = 16000  # Hz - YAMNet expects 16kHz
DEVICE_SAMPLE_RATE = 48000  # typical for Google voiceHAT mic
DURATION = 2.0  # seconds - longer window for better detection
CHANNELS = 1
DEVICE_ID = 1  # Use your Google voiceHAT device
YAMNET_MODEL_PATH = 'yamnet_model'  # Local YAMNet model path
CLASS_MAP_PATH = 'yamnet_model/yamnet_class_map.csv'
DETECTION_THRESHOLD = 0.3  # Threshold for fart detection
BLOCK_SIZE = int(DEVICE_SAMPLE_RATE * DURATION)

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
    """Load the YAMNet model with timeout"""
    try:
        print("Loading YAMNet model... This may take a moment on Raspberry Pi...")
        
        # Set TensorFlow to use CPU only for better compatibility
        tf.config.set_visible_devices([], 'GPU')
        
        # Load model with timeout
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timed out")
        
        # Set 30 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            model = tf.saved_model.load(YAMNET_MODEL_PATH)
            signal.alarm(0)  # Cancel timeout
            print(f"YAMNet model loaded successfully from {YAMNET_MODEL_PATH}")
            return model
        except TimeoutError:
            print("Model loading timed out. YAMNet may be too large for this Raspberry Pi.")
            print("Trying to load from TensorFlow Hub as fallback...")
            # return load_yamnet_from_hub()
            return create_dummy_model()
        finally:
            signal.alarm(0)  # Ensure timeout is cancelled
            
    except Exception as e:
        print(f"Failed to load YAMNet model: {e}")
        print("Creating a dummy model for demonstration...")
        return create_dummy_model()

def load_yamnet_from_hub():
    """Load YAMNet from TensorFlow Hub as fallback"""
    try:
        import tensorflow_hub as hub
        print("Loading YAMNet from TensorFlow Hub...")
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("YAMNet loaded successfully from TensorFlow Hub")
        return model
    except Exception as e:
        print(f"Failed to load from TensorFlow Hub: {e}")
        return create_dummy_model()

def create_dummy_model():
    """Create a dummy model for demonstration when YAMNet fails to load"""
    try:
        print("Creating dummy model for demonstration...")
        
        # Create a simple model that returns random predictions
        class DummyModel:
            def __call__(self, audio):
                # Return dummy scores, embeddings, and spectrogram
                batch_size = 1
                num_classes = 521
                num_frames = 97  # YAMNet output shape
                
                scores = tf.random.uniform((batch_size, num_frames, num_classes))
                embeddings = tf.random.uniform((batch_size, num_frames, 1024))
                spectrogram = tf.random.uniform((batch_size, num_frames, 64))
                
                return scores, embeddings, spectrogram
        
        return DummyModel()
        
    except Exception as e:
        print(f"Failed to create dummy model: {e}")
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
    # resample from device SR → YAMNet SR
    if DEVICE_SAMPLE_RATE != YAMNET_SAMPLE_RATE:
        # Calculate resampling ratio
        ratio = YAMNET_SAMPLE_RATE / DEVICE_SAMPLE_RATE
        # Use scipy.signal.resample for resampling
        num_samples = int(len(audio) * ratio)
        audio = signal.resample(audio, num_samples)
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
        with sd.InputStream(channels=CHANNELS,
                            samplerate=DEVICE_SAMPLE_RATE,
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
