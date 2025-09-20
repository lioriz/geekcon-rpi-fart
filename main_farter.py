#!/usr/bin/env python3
"""
Farter Robot Main Program
Drives around randomly and plays fart sounds at random intervals
"""

import subprocess
import glob
import time
import random
import os
import threading
import logging
from datetime import datetime
from motor_control import MotorController
from logger_utils import get_logger

# Configuration
MP3_DIR = os.path.expanduser("~/farts/*.mp3")  # Directory containing mp3 files
VOLUME = 10  # Playback volume (0–32768, default 32768 = 100%)
SLEEP_BETWEEN_SOUNDS = (2, 8)  # Random sleep between sounds (min, max) in seconds
SLEEP_BETWEEN_MOVEMENTS = (1, 5)  # Random sleep between movements (min, max) in seconds

# Motor settings
MOTOR_SERIAL_PORT = '/dev/serial0'  # Serial port for motor communication
MOTOR_SPEED_RANGE = (-500, 500)  # Random motor speed range (min, max)
MOTOR_DURATION = (1, 5)  # Random movement duration (min, max) in seconds

# Initialize logger
logger = get_logger('farter_robot', logging.INFO)

class FarterRobot:
    """Main class for the farter robot"""
    
    def __init__(self):
        self.motor_controller = None
        self.sound_files = []
        self.running = False
        self.stop_event = threading.Event()
        
    def initialize(self):
        """Initialize the robot components"""
        logger.info("🤖 Initializing Farter Robot...")
        
        # Find all mp3 files
        self.sound_files = glob.glob(MP3_DIR)
        if not self.sound_files:
            logger.error(f"No MP3 files found in {MP3_DIR}")
            return False
        
        logger.info(f"Found {len(self.sound_files)} sound files")
        
        # Initialize motor controller
        try:
            self.motor_controller = MotorController(port=MOTOR_SERIAL_PORT)
            if not self.motor_controller.connect():
                logger.error("Failed to connect to motor controller")
                return False
            
            # Test motor communication
            if not self.motor_controller.ping():
                logger.warning("Motor controller not responsive, continuing anyway")
            
            logger.info("🤖 Motor controller initialized")
            
        except Exception as e:
            logger.error(f"Motor controller initialization failed: {e}")
            return False
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")
        self.running = False
        self.stop_event.set()
        
        if self.motor_controller:
            try:
                self.motor_controller.stop_motors()
                self.motor_controller.disconnect()
                logger.info("🤖 Motor controller disconnected")
            except Exception as e:
                logger.error(f"Motor cleanup error: {e}")
    
    def play_random_sound(self):
        """Play a random fart sound"""
        if not self.sound_files:
            return
        
        sound_file = random.choice(self.sound_files)
        logger.info(f"🔊 Playing: {os.path.basename(sound_file)}")
        
        try:
            subprocess.run([
                "mpg123", "-q", "-o", "alsa", 
                "-f", str(VOLUME), sound_file
            ], check=True)
            logger.info("🔊 Sound playback complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to play sound: {e}")
        except FileNotFoundError:
            logger.error("mpg123 not found. Please install it: sudo apt install mpg123")
    
    def random_movement(self):
        """Perform a random movement with random left and right motor speeds"""
        if not self.motor_controller:
            return
        
        # Generate random speeds for left and right motors
        left_speed = random.randint(*MOTOR_SPEED_RANGE)
        right_speed = random.randint(*MOTOR_SPEED_RANGE)
        duration = random.uniform(*MOTOR_DURATION)
        
        logger.info(f"🤖 Random movement: L={left_speed}, R={right_speed} for {duration:.1f}s")
        
        try:
            # Set random motor speeds
            self.motor_controller.set_motors(left_speed, right_speed)
            
            # Wait for movement duration
            time.sleep(duration)
            
            # Stop motors
            self.motor_controller.stop_motors()
            logger.info("🤖 Movement complete")
            
        except Exception as e:
            logger.error(f"Movement error: {e}")
            try:
                self.motor_controller.stop_motors()
            except:
                pass
    
    def sound_worker(self):
        """Worker thread for playing sounds"""
        logger.info("🔊 Sound worker started")
        
        while not self.stop_event.is_set():
            try:
                # Play a random sound
                self.play_random_sound()
                
                # Random sleep between sounds
                sleep_time = random.uniform(*SLEEP_BETWEEN_SOUNDS)
                logger.info(f"🔊 Next sound in {sleep_time:.1f}s")
                
                # Sleep with interrupt capability
                if self.stop_event.wait(sleep_time):
                    break
                    
            except Exception as e:
                logger.error(f"Sound worker error: {e}")
                time.sleep(1)
        
        logger.info("🔊 Sound worker stopped")
    
    def movement_worker(self):
        """Worker thread for random movements"""
        logger.info("🤖 Movement worker started")
        
        while not self.stop_event.is_set():
            try:
                # Perform random movement
                self.random_movement()
                
                # Random sleep between movements
                sleep_time = random.uniform(*SLEEP_BETWEEN_MOVEMENTS)
                logger.info(f"🤖 Next movement in {sleep_time:.1f}s")
                
                # Sleep with interrupt capability
                if self.stop_event.wait(sleep_time):
                    break
                    
            except Exception as e:
                logger.error(f"Movement worker error: {e}")
                time.sleep(1)
        
        logger.info("🤖 Movement worker stopped")
    
    def run(self):
        """Main run loop"""
        if not self.initialize():
            logger.error("Failed to initialize robot")
            return
        
        self.running = True
        logger.info("=" * 50)
        logger.info("🤖 Farter Robot Started!")
        logger.info("=" * 50)
        logger.info(f"Sound files: {len(self.sound_files)}")
        logger.info(f"Sound interval: {SLEEP_BETWEEN_SOUNDS[0]}-{SLEEP_BETWEEN_SOUNDS[1]}s")
        logger.info(f"Movement interval: {SLEEP_BETWEEN_MOVEMENTS[0]}-{SLEEP_BETWEEN_MOVEMENTS[1]}s")
        logger.info("Press Ctrl+C to stop")
        
        try:
            # Start worker threads
            sound_thread = threading.Thread(target=self.sound_worker, name="SoundWorker")
            movement_thread = threading.Thread(target=self.movement_worker, name="MovementWorker")
            
            sound_thread.start()
            movement_thread.start()
            
            # Keep main thread alive
            while self.running and not self.stop_event.is_set():
                time.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping robot...")
            self.stop_event.set()
            
            # Wait for threads to finish
            sound_thread.join(timeout=2)
            movement_thread.join(timeout=2)
            
        finally:
            self.cleanup()
            logger.info("✅ Farter Robot stopped")


def main():
    """Main entry point"""
    robot = FarterRobot()
    robot.run()


if __name__ == "__main__":
    main()
