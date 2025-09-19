#!/usr/bin/env python3
"""
Simple Raspberry Pi Fart Detector using YAMNet
Continuously listens to USB microphone and detects fart sounds
"""

import sounddevice as sd
import numpy as np
import csv
import time
from datetime import datetime
from scipy import signal  # for resampling
import threading
from collections import deque
import queue
import logging
import os
import wave
from motor_control import MotorController
from logger_utils import get_logger

# Global configuration constants
YAMNET_SAMPLE_RATE = 16000  # Hz - YAMNet expects 16kHz
DEVICE_SAMPLE_RATE = 48000  # typical for Google voiceHAT mic
DURATION = 1.0  # seconds - longer window for better detection
CHANNELS = 2  # Stereo - use both microphones
DEVICE_ID = 1  # Use your Google voiceHAT device
MODEL_PATH = 'yamnet_model/yamnet.tflite'  # TensorFlow Lite model path
CLASS_MAP_PATH = 'yamnet_model/yamnet_class_map.csv'

# Detection settings - adjust these for better sensitivity
DETECTION_THRESHOLD = 0.003  # Lower = more sensitive
MIC_DISTANCE = 0.08  # Distance between microphones in meters
BLOCK_SIZE = int(DEVICE_SAMPLE_RATE * DURATION)

# Threading settings
AUDIO_QUEUE_SIZE = 10  # Number of audio chunks to buffer
PROCESSING_THREADS = 1  # Number of processing threads

# Recording save settings
SAVE_RECORDINGS = False  # Enable/disable recording save feature
SAVE_EVERY_N_RECORDINGS = 10  # Save every 10th recording
RECORDINGS_DIR = "recordings"

# Motor control settings
MOTOR_ENABLED = True  # Enable/disable motor control
MOTOR_DRIVE_STEPS = 10  # Steps to drive forward after detection
MOTOR_ROTATION_STEPS_PER_DEGREE = 5 / 90  # Steps per degree of rotation (5 steps = 90 degrees)
MOTOR_SERIAL_PORT = '/dev/serial0'  # Serial port for motor communication
MOTOR_QUEUE_SIZE = 10  # Max motor commands to queue
MOTOR_COOLDOWN = 3.0  # Seconds to wait between motor responses

# Initialize logger
logger = get_logger('fart_detector', logging.INFO)

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
        logger.info(f"Fart class indices: {fart_indices}")
        return class_map, fart_indices
    except Exception as e:
        logger.error(f"Failed to load class map: {e}")
        return {}, []

def load_tflite_model():
    """Load the TensorFlow Lite model with XNNPACK optimization"""
    try:
        import tflite_runtime.interpreter as tflite

        # XNNPACK is built-in; enable by setting num_threads
        interpreter = tflite.Interpreter(
            model_path=MODEL_PATH,
            num_threads=4  # Use all 4 cores of Raspberry Pi 4
        )
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        logger.info(f"TensorFlow Lite model loaded successfully from {MODEL_PATH}")
        logger.info(f"XNNPACK optimization enabled (threads=4)")
        logger.info(f"Input shape: {input_details[0]['shape']}")
        logger.info(f"Output shape: {output_details[0]['shape']}")
        
        return interpreter, input_details, output_details
        
    except Exception as e:
        logger.error(f"Failed to load TensorFlow Lite model: {e}")
        logger.warning("Creating a dummy model for demonstration...")
        return create_dummy_model()

def create_dummy_model():
    """Create a dummy model for demonstration when TFLite model fails to load"""
    try:
        logger.warning("Creating dummy model for demonstration...")
        
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
        logger.error(f"Failed to create dummy model: {e}")
        return None, None, None

def estimate_direction(left, right, samplerate, mic_distance=0.08):
    """
    Estimate direction of arrival from two microphones using TDOA.
    - left, right: audio arrays
    - samplerate: sample rate in Hz
    - mic_distance: spacing between mics in meters (8cm default)
    Returns angle in degrees (negative = left, positive = right).
    """
    # Downsample for faster correlation (much faster!)
    downsample_factor = 4
    left_ds = left[::downsample_factor]
    right_ds = right[::downsample_factor]
    
    # Cross-correlation on downsampled data
    corr = np.correlate(left_ds, right_ds, mode='full')
    delay_samples = np.argmax(corr) - (len(left_ds) - 1)
    delay_time = (delay_samples * downsample_factor) / samplerate

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

def save_audio_to_wav(audio, filename, sample_rate):
    """Save audio data to WAV file"""
    try:
        # Ensure audio is in the right format (16-bit PCM)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        logger.info(f"💾 Saved recording: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to save recording {filename}: {e}")
        return False

def recording_saver(save_queue, stop_event):
    """Thread function to save recordings to disk"""
    logger.info("💾 Recording saver started")
    
    # Create recordings directory
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    
    while not stop_event.is_set():
        try:
            # Get recording from queue (with timeout)
            recording_data = save_queue.get(timeout=1.0)
            
            audio_data = recording_data['audio']
            filename = recording_data['filename']
            sample_rate = recording_data['sample_rate']
            reason = recording_data.get('reason', 'unknown')
            
            # Save the recording
            success = save_audio_to_wav(audio_data, filename, sample_rate)
            
            if success:
                logger.info(f"💾 Saved {reason} recording: {filename}")
            else:
                logger.error(f"❌ Failed to save {reason} recording: {filename}")
            
            # Mark task as done only if we successfully processed an item
            save_queue.task_done()
                
        except queue.Empty:
            continue  # Timeout, check stop_event
        except Exception as e:
            logger.error(f"❌ Recording saver error: {e}")
            # Mark task as done even if processing failed
            save_queue.task_done()
        # Only call task_done() if we actually got an item from the queue
    
    logger.info("💾 Recording saver stopped")

def motor_command_processor(motor_queue, motor_controller, stop_event):
    """Dedicated thread to process motor commands sequentially"""
    logger.info("🤖 Motor command processor started")
    last_motor_time = 0
    
    while not stop_event.is_set():
        try:
            # Get motor command from queue (with timeout)
            command = motor_queue.get(timeout=1.0)
            
            # Check cooldown period
            current_time = time.time()
            if current_time - last_motor_time < MOTOR_COOLDOWN:
                logger.info(f"🤖 Motor cooldown active, skipping command")
                motor_queue.task_done()
                continue
            
            angle = command['angle']
            logger.info(f"🤖 Processing motor command: {angle:+.1f}°")
            
            # Execute motor movement
            execute_motor_movement(angle, motor_controller)
            
            last_motor_time = current_time
            motor_queue.task_done()
            
        except queue.Empty:
            continue  # Timeout, check stop_event
        except Exception as e:
            logger.error(f"❌ Motor command processor error: {e}")
            try:
                motor_queue.task_done()
            except:
                pass
    
    logger.info("🤖 Motor command processor stopped")

def execute_motor_movement(angle, motor_controller):
    """Execute the actual motor movement (extracted from thread)"""
    try:
        # Calculate rotation steps based on angle
        if abs(angle) < 10:
            # Fart is in front, just drive forward
            logger.info("🤖 Fart in front, driving forward")
            motor_controller.move_steps(MOTOR_DRIVE_STEPS, MOTOR_DRIVE_STEPS)
        else:
            # Fart is to the side, rotate first then drive
            rotation_steps = int(abs(angle) * MOTOR_ROTATION_STEPS_PER_DEGREE)
            
            if angle > 0:
                # Fart is to the right, rotate right
                logger.info(f"🤖 Rotating right {rotation_steps} steps")
                motor_controller.move_steps(-rotation_steps, rotation_steps)
            else:
                # Fart is to the left, rotate left
                logger.info(f"🤖 Rotating left {rotation_steps} steps")
                motor_controller.move_steps(-rotation_steps, rotation_steps)
            
            # Wait for rotation to complete
            time.sleep(0.5)  # Brief pause between rotation and drive
            
            # Drive forward
            logger.info(f"🤖 Driving forward {MOTOR_DRIVE_STEPS} steps")
            motor_controller.move_steps(MOTOR_DRIVE_STEPS, MOTOR_DRIVE_STEPS)
        
        logger.info("🤖 Motor movement complete")
        
    except Exception as e:
        logger.error(f"❌ Motor movement error: {e}")
        try:
            motor_controller.stop_motors()
        except:
            pass


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

def audio_producer(audio_queue, stop_event):
    """Producer thread: captures audio and puts it in queue"""
    logger.info("🎤 Audio producer started")
    
    try:
        with sd.InputStream(channels=CHANNELS,
                            samplerate=DEVICE_SAMPLE_RATE,
                            blocksize=BLOCK_SIZE,
                            dtype='float32',
                            device=DEVICE_ID) as stream:
            while not stop_event.is_set():
                start_time = time.time()
                # Read audio block from stream (stereo)
                audio, overflowed = stream.read(BLOCK_SIZE)
                
                # Put audio in queue (non-blocking)
                try:
                    audio_queue.put_nowait(audio)
                    #logger.info(f"🎤 Audio producer: {time.time() - start_time:.3f}s")
                except queue.Full:
                    logger.warning("⚠️  Audio queue full, dropping frame")
                    
    except Exception as e:
        logger.error(f"❌ Audio producer error: {e}")
    finally:
        logger.info("🎤 Audio producer stopped")

def audio_processor(audio_queue, save_queue, motor_queue, interpreter, input_details, output_details, 
                   fart_indices, class_map, stop_event, motor_controller=None):
    """Consumer thread: processes audio from queue"""
    logger.info(f"🔧 Audio processor started")
    recording_counter = 0
    
    while not stop_event.is_set():
        try:
            # Get audio from queue (with timeout)
            audio = audio_queue.get(timeout=2.0)
            
            # Split into left and right channels
            left_channel = audio[:, 0]   # Left microphone
            right_channel = audio[:, 1]  # Right microphone
            
            direction_start = time.time()
            # Estimate direction of arrival
            angle, delay = estimate_direction(left_channel, right_channel, DEVICE_SAMPLE_RATE, MIC_DISTANCE)
            
            preprocess_start = time.time()
            # Process both channels
            processed_left = preprocess_audio(left_channel)
            processed_right = preprocess_audio(right_channel)
            
            # Run inference on both channels
            left_inference_start = time.time()
            conf_left = predict_fart(interpreter, input_details, output_details, processed_left, fart_indices, class_map, "L")
            left_inference_time = time.time()
            
            right_inference_start = time.time()
            conf_right = predict_fart(interpreter, input_details, output_details, processed_right, fart_indices, class_map, "R")
            right_inference_time = time.time()
            
            # Calculate levels
            level_left = np.sqrt(np.mean(processed_left**2))
            level_right = np.sqrt(np.mean(processed_right**2))
            
            # Detection logic
            max_confidence = max(conf_left, conf_right)
            avg_level = (level_left + level_right) / 2
            
            # Timing analysis
            total_time = time.time() - direction_start
            direction_duration = preprocess_start - direction_start
            preprocess_duration = left_inference_start - preprocess_start
            left_inference_duration = left_inference_time - left_inference_start
            right_inference_duration = right_inference_time - right_inference_start
            
            # Log results
            #logger.info(f"🧭 {angle:+.1f}° | L: confidance-{conf_left:.3f}, level-{level_left:.3f} | R: confidance-{conf_right:.3f}, level-{level_right:.3f} | ⏱️ Total={total_time:.3f}s | Direction={direction_duration:.3f}s | Preprocess={preprocess_duration:.3f}s | L_inf={left_inference_duration:.3f}s | R_inf={right_inference_duration:.3f}s")
            
            # Detection logic (always runs)
            if max_confidence >= DETECTION_THRESHOLD:
                # Determine direction arrow
                if abs(angle) < 10:
                    direction_arrow = "⬆️"  # Front
                elif angle > 0:
                    direction_arrow = "➡️"  # Right
                else:
                    direction_arrow = "⬅️"  # Left
                
                logger.info(f"🚨🚨🚨🚨 FART DETECTED! {direction_arrow}, conf: {max_confidence:.3f}, level: {avg_level:.3f} direction: {angle:+.1f}° 🚨🚨🚨🚨")
                
                # Queue motor response if enabled
                if MOTOR_ENABLED and motor_controller is not None and motor_queue is not None:
                    try:
                        # Add motor command to queue
                        motor_queue.put_nowait({
                            'angle': angle,
                            'timestamp': time.time()
                        })
                        logger.info(f"🤖 Motor command queued: {angle:+.1f}°")
                    except queue.Full:
                        logger.warning("⚠️ Motor queue full, dropping command")
                    except Exception as e:
                        logger.error(f"❌ Failed to queue motor response: {e}")
            # else:
                #logger.info(f"... quiet (max conf: {max_confidence:.3f}, avg level: {avg_level:.3f})")
            
            if SAVE_RECORDINGS:
                # Increment recording counter
                recording_counter += 1
                # Check if we should save this recording
                should_save = False
                save_reason = ""
                if max_confidence >= DETECTION_THRESHOLD:
                    # Save detected fart
                    should_save = True
                    save_reason = "fart_detected"
                elif recording_counter % SAVE_EVERY_N_RECORDINGS == 0:
                    # Save every Nth recording
                    should_save = True
                    save_reason = f"every_{SAVE_EVERY_N_RECORDINGS}"
            
                # Queue recording for saving if needed
                if should_save:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"{RECORDINGS_DIR}/{save_reason}_{timestamp}.wav"
                    
                    try:
                        save_queue.put_nowait({
                            'audio': audio,
                            'filename': filename,
                            'sample_rate': DEVICE_SAMPLE_RATE,
                            'reason': save_reason
                        })
                        logger.info(f"💾 Queued {save_reason} recording: {filename}")
                    except queue.Full:
                        logger.warning("⚠️ Save queue full, dropping recording")
            
            
            # Mark task as done only if we successfully processed an item
            audio_queue.task_done()
 
        except queue.Empty:
            logger.warning(f"Timeout")
            continue  # Timeout, check stop_event
        except Exception as e:
            logger.error(f"❌ Processor error: {e}")
            # Mark task as done even if processing failed
            audio_queue.task_done()
    
    logger.info(f"🔧 Audio processor stopped")

def predict_fart(interpreter, input_details, output_details, audio, fart_indices, class_map, mic_id="Unknown"):
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
        
        #logger.info(f"Top {top_predictions} ({mic_id}): " + ", ".join([f"{class_map.get(idx, f'Unknown_{idx}')}: {scores[idx]:.4f}" for idx in top_indices]))
        
        # Check fart probability
        max_fart_score = 0.0
        for idx in fart_indices:
            fart_score = mean_scores[0][idx]
            if fart_score > 0.0:
                class_name = class_map.get(idx, f"Unknown_{idx}")
                logger.info(f"💩💨 ({mic_id}): fart detected: \"{class_name}\", with score: {fart_score:.4f} 💩💨")
            max_fart_score = max(max_fart_score, fart_score)
        
        return max_fart_score
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 0.0

def main():
    """Main detection loop using multi-threaded TensorFlow Lite"""
    logger.info("=" * 50)
    logger.info("Multi-threaded Raspberry Pi Fart Detector")
    logger.info("=" * 50)
    
    # List available audio devices
    try:
        devices = sd.query_devices()
        logger.info(f"Available audio devices: {len(devices)}")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                logger.info(f"  Device {i}: {device['name']} (inputs: {device['max_input_channels']})")
    except Exception as e:
        logger.error(f"Failed to query audio devices: {e}")
    
    # Load class map and find fart indices
    class_map, fart_indices = load_class_map()
    
    # Load TensorFlow Lite model
    interpreter, input_details, output_details = load_tflite_model()
    
    if interpreter is None:
        logger.error("Failed to load TensorFlow Lite model. Exiting.")
        return
    
    # Initialize motor controller if enabled
    motor_controller = None
    if MOTOR_ENABLED:
        try:
            motor_controller = MotorController(port=MOTOR_SERIAL_PORT)
            if motor_controller.connect():
                logger.info("🤖 Motor controller connected successfully")
                # Test motor communication
                if motor_controller.ping():
                    logger.info("🤖 Motor controller is responsive")
                else:
                    logger.warning("🤖 Motor controller not responsive, continuing anyway")
            else:
                logger.error("🤖 Failed to connect to motor controller")
                motor_controller = None
        except Exception as e:
            logger.error(f"🤖 Motor controller initialization failed: {e}")
            motor_controller = None
    
    logger.info(f"Starting multi-threaded detection...")
    logger.info(f"Device sample rate: {DEVICE_SAMPLE_RATE} Hz")
    logger.info(f"YAMNet sample rate: {YAMNET_SAMPLE_RATE} Hz")
    logger.info(f"Duration: {DURATION} seconds")
    logger.info(f"Threshold: {DETECTION_THRESHOLD}")
    logger.info(f"Processing threads: {PROCESSING_THREADS}")
    logger.info("Press Ctrl+C to stop")
    
    # Create shared resources
    audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
    save_queue = queue.Queue(maxsize=50)  # Queue for recordings to save
    motor_queue = queue.Queue(maxsize=MOTOR_QUEUE_SIZE)  # Queue for motor commands
    stop_event = threading.Event()
    
    # Start producer thread (audio capture)
    producer_thread = threading.Thread(
        target=audio_producer,
        args=(audio_queue, stop_event),
        name="🎤AudioCapture"
    )
    producer_thread.start()
    
    # Start recording saver thread (only if saving is enabled)
    saver_thread = None
    if SAVE_RECORDINGS:
        saver_thread = threading.Thread(
            target=recording_saver,
            args=(save_queue, stop_event),
            name="💾RecordingSaver"
        )
        saver_thread.start()
    
    # Start motor command processor thread (only if motor is enabled)
    motor_processor_thread = None
    if MOTOR_ENABLED and motor_controller is not None:
        motor_processor_thread = threading.Thread(
            target=motor_command_processor,
            args=(motor_queue, motor_controller, stop_event),
            name="🤖MotorProcessor"
        )
        motor_processor_thread.start()
    
    # Start consumer threads (audio processing)
    processor_threads = []
    for i in range(PROCESSING_THREADS):
        thread = threading.Thread(
            target=audio_processor,
            args=(audio_queue, save_queue, motor_queue, interpreter, input_details, output_details, 
                  fart_indices, class_map, stop_event, motor_controller),
            name=f"🔧AudioProcess-{i+1}"
        )
        thread.start()
        processor_threads.append(thread)
    
    try:
        logger.info("🎤 All threads started, listening for farts...")
        # Keep main thread alive
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping all threads...")
        stop_event.set()
        
        # Wait for threads to finish
        producer_thread.join(timeout=2)
        for thread in processor_threads:
            thread.join(timeout=2)
        if saver_thread:
            saver_thread.join(timeout=2)
        if motor_processor_thread:
            motor_processor_thread.join(timeout=2)
        
        logger.info("✅ All threads stopped")
        
        # Stop motors and disconnect
        if motor_controller:
            try:
                motor_controller.stop_motors()
                motor_controller.disconnect()
                logger.info("🤖 Motor controller disconnected")
            except Exception as e:
                logger.error(f"🤖 Motor cleanup error: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        stop_event.set()
        
        # Emergency motor stop
        if motor_controller:
            try:
                motor_controller.stop_motors()
                motor_controller.disconnect()
            except:
                pass

if __name__ == "__main__":
    main()
