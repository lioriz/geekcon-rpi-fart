#!/usr/bin/env python3
"""
Simple Raspberry Pi Fart Detector using YAMNet
Continuously listens to USB microphone and detects fart sounds
"""

import sounddevice as sd
import numpy as np
import csv
import time
from scipy import signal  # for resampling

# Global configuration constants
YAMNET_SAMPLE_RATE = 16000  # Hz - YAMNet expects 16kHz
DEVICE_SAMPLE_RATE = 48000  # typical for Google voiceHAT mic
DURATION = 1.0  # seconds - longer window for better detection
CHANNELS = 2  # Stereo - use both microphones
DEVICE_ID = 1  # Use your Google voiceHAT device
MODEL_PATH = 'yamnet_model/yamnet.tflite'  # TensorFlow Lite model path
CLASS_MAP_PATH = 'yamnet_model/yamnet_class_map.csv'

# Detection settings - adjust these for better sensitivity
DETECTION_THRESHOLD = 0.001  # Lower = more sensitive
MIC_DISTANCE = 0.08  # Distance between microphones in meters
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

def load_tflite_model():
    """Load the TensorFlow Lite model"""
    try:
        import tflite_runtime.interpreter as tflite
        
        # Load the TFLite model
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"TensorFlow Lite model loaded successfully from {MODEL_PATH}")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Output shape: {output_details[0]['shape']}")
        
        return interpreter, input_details, output_details
        
    except Exception as e:
        print(f"Failed to load TensorFlow Lite model: {e}")
        print("Creating a dummy model for demonstration...")
        return create_dummy_model()

def create_dummy_model():
    """Create a dummy model for demonstration when TFLite model fails to load"""
    try:
        print("Creating dummy model for demonstration...")
        
        # Create a simple model that returns random predictions
        class DummyModel:
            def __init__(self):
                self.input_details = [{'shape': [1, 15600]}]
                self.output_details = [{'shape': [1, 97, 521]}]
            
            def allocate_tensors(self):
                pass
            
            def set_tensor(self, index, value):
                pass
            
            def invoke(self):
                pass
            
            def get_tensor(self, index):
                # Return dummy scores
                return np.random.random((1, 97, 521)).astype(np.float32)
        
        return DummyModel(), None, None
        
    except Exception as e:
        print(f"Failed to create dummy model: {e}")
        return None, None, None

def estimate_direction(left, right, samplerate, mic_distance=0.08):
    """
    Estimate direction of arrival from two microphones using TDOA.
    - left, right: audio arrays
    - samplerate: sample rate in Hz
    - mic_distance: spacing between mics in meters (8cm default)
    Returns angle in degrees (negative = left, positive = right).
    """
    # Cross-correlation to find time delay
    corr = np.correlate(left, right, mode='full')
    delay_samples = np.argmax(corr) - (len(left) - 1)
    delay_time = delay_samples / samplerate

    # Convert delay to angle
    c = 343.0  # speed of sound (m/s)
    value = (delay_time * c) / mic_distance
    # Clamp value to [-1, 1] to avoid math domain error
    value = max(-1.0, min(1.0, value))
    angle_rad = np.arcsin(value)
    angle_deg = np.degrees(angle_rad)

    return angle_deg, delay_time

def apply_gain(audio, target_rms=0.1, max_gain=10.0):
    """
    Apply automatic software gain to audio.
    - target_rms: desired RMS level (0.1 is ~10% of full scale)
    - max_gain: cap on gain factor to avoid blowing up noise
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        gain = min(max_gain, target_rms / rms)
        audio = audio * gain
        # Clip to [-1, 1] to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)
    return audio

def preprocess_audio(audio):
    """Preprocess audio for YAMNet model"""
    # Ensure audio is mono and float32
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    # Apply software gain
    audio = apply_gain(audio, target_rms=0.1, max_gain=10.0)

    # Pre-emphasis filter (optional, helps with bursts)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Resample from device SR → YAMNet SR
    if DEVICE_SAMPLE_RATE != YAMNET_SAMPLE_RATE:
        ratio = YAMNET_SAMPLE_RATE / DEVICE_SAMPLE_RATE
        num_samples = int(len(audio) * ratio)
        audio = signal.resample(audio, num_samples)

    return audio

def predict_fart(interpreter, input_details, output_details, audio, fart_indices, class_map):
    """Predict if audio contains a fart using TensorFlow Lite"""
    if interpreter is None:
        # Return random prediction for demonstration
        return np.random.random()
    
    try:
        # Ensure audio is the right length (YAMNet expects exactly 15600 samples)
        target_length = 15600  # YAMNet input length
        if len(audio) != target_length:
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                # Truncate
                audio = audio[:target_length]
        
        # Model expects 1D array, not 2D
        input_data = audio.astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Output is already in the right shape: (1, 521)
        mean_scores = output_data  # Shape: (1, 521)
        
        # Get top predictions with names
        top_predictions = 2
        scores = mean_scores[0]  # Shape: (521,)
        top_indices = np.argsort(scores)[-top_predictions:][::-1]  # Top, highest first
        
        print(f"Top {top_predictions}: " + ", ".join([f"{class_map.get(idx, f'Unknown_{idx}')}: {scores[idx]:.4f}" for idx in top_indices]))
        
        # Check fart probability
        max_fart_score = 0.0
        for idx in fart_indices:
            fart_score = mean_scores[0][idx]
            if fart_score > 0.0:
                class_name = class_map.get(idx, f"Unknown_{idx}")
                print(f"💩💨 Fart detected: \"{class_name}\", with score: {fart_score:.4f} 💩💨")
            max_fart_score = max(max_fart_score, fart_score)
        
        return max_fart_score
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return 0.0

def main():
    """Main detection loop using TensorFlow Lite"""
    print("=" * 50)
    print("Simple Raspberry Pi Fart Detector using TensorFlow Lite")
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
    
    # Load TensorFlow Lite model
    interpreter, input_details, output_details = load_tflite_model()
    
    if interpreter is None:
        print("Failed to load TensorFlow Lite model. Exiting.")
        return
    
    print(f"Starting detection loop...")
    print(f"Device sample rate: {DEVICE_SAMPLE_RATE} Hz")
    print(f"YAMNet sample rate: {YAMNET_SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print(f"Threshold: {DETECTION_THRESHOLD}")
    print("Press Ctrl+C to stop")
    
    try:
        with sd.InputStream(channels=CHANNELS,
                            samplerate=DEVICE_SAMPLE_RATE,
                            blocksize=BLOCK_SIZE,
                            dtype='float32',
                            device=DEVICE_ID) as stream:
            print("Listening for farts...")
            
            while True:
                # Read audio block from stream (stereo)
                audio, overflowed = stream.read(BLOCK_SIZE)
                
                # Split into left and right channels
                left_channel = audio[:, 0]   # Left microphone
                right_channel = audio[:, 1]  # Right microphone
                
                # Estimate direction of arrival
                angle, delay = estimate_direction(left_channel, right_channel, DEVICE_SAMPLE_RATE, MIC_DISTANCE)
                
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
                print(f"🧭 Direction: {angle:+.1f}° (delay: {delay*1e6:.1f} µs)")
                
                # Process LEFT channel
                print("🔴 LEFT MICROPHONE:")
                processed_left = preprocess_audio(left_channel)
                conf_left = predict_fart(interpreter, input_details, output_details, processed_left, fart_indices, class_map)
                level_left = np.sqrt(np.mean(processed_left**2))
                
                # Process RIGHT channel  
                print("🔵 RIGHT MICROPHONE:")
                processed_right = preprocess_audio(right_channel)
                conf_right = predict_fart(interpreter, input_details, output_details, processed_right, fart_indices, class_map)
                level_right = np.sqrt(np.mean(processed_right**2))
                
                # Combined detection results
                print(f"📊 L: conf={conf_left:.3f}, level={level_left:.3f}, R: conf={conf_right:.3f}, level={level_right:.3f}")
                
                # Detection logic - either mic can detect
                max_confidence = max(conf_left, conf_right)
                avg_level = (level_left + level_right) / 2
                
                if max_confidence >= DETECTION_THRESHOLD:
                    # Determine direction arrow
                    if abs(angle) < 10:
                        direction_arrow = "⬆️"  # Front
                    elif angle > 0:
                        direction_arrow = "➡️"  # Right
                    else:
                        direction_arrow = "⬅️"  # Left
                    
                    print(f"🚨🚨🚨🚨 FART DETECTED! {direction_arrow}, conf: {max_confidence:.3f}, level: {avg_level:.3f} direction: {angle:+.1f}° 🚨🚨🚨🚨")
                else:
                    print(f"... quiet (max conf: {max_confidence:.3f}, avg level: {avg_level:.3f})")
                print("=" * 60)
                
    except KeyboardInterrupt:
        print("\nStopping detection...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
