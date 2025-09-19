#!/usr/bin/env python3
"""
Motor Control Module for Geekcon Robot
Communicates with STM32L412xx MCU over USART1 using binary protocol
"""

import serial
import struct
import time
import logging

# Protocol constants
SOF = 0xAA  # Start of Frame
EOF = 0x55  # End of Frame

# Command IDs (Raspberry Pi -> MCU)
CMD_SET_MOTORS = 0x01
CMD_GET_ENCODERS = 0x02
CMD_RESET_ENCODERS = 0x03
CMD_PING = 0x04
CMD_MOVE_STEPS = 0x05

# Message IDs (MCU -> Raspberry Pi)
MSG_ENCODER_DATA = 0x11
MSG_ACK = 0x12
MSG_PONG = 0x13
MSG_ERROR = 0xEE

class MotorController:
    """Motor controller for Geekcon Robot MCU communication"""
    
    def __init__(self, port='/dev/serial0', baudrate=115200, timeout=1.0):
        """
        Initialize motor controller
        
        Args:
            port: Serial port (default: /dev/serial0 for Raspberry Pi)
            baudrate: Baud rate (default: 115200)
            timeout: Serial timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.logger = logging.getLogger('motor_control')
        
    def connect(self):
        """Connect to the MCU"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.logger.info(f"Connected to MCU on {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to MCU: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the MCU"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.logger.info("Disconnected from MCU")
    
    def _calculate_checksum(self, command_id, payload):
        """Calculate XOR checksum for message"""
        checksum = command_id ^ len(payload)
        for byte in payload:
            checksum ^= byte
        return checksum & 0xFF
    
    def _send_message(self, command_id, payload=b''):
        """
        Send a message to the MCU
        
        Args:
            command_id: Command/Message ID
            payload: Payload bytes
            
        Returns:
            bool: True if message sent successfully
        """
        if not self.ser or not self.ser.is_open:
            self.logger.error("Not connected to MCU")
            return False
        
        try:
            # Calculate checksum
            checksum = self._calculate_checksum(command_id, payload)
            
            # Construct message
            message = bytearray([SOF, command_id, len(payload)])
            message.extend(payload)
            message.append(checksum)
            message.append(EOF)
            
            # Send message
            self.ser.write(message)
            self.logger.debug(f"Sent: {' '.join(f'{b:02X}' for b in message)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    def _read_response(self, expected_id=None, timeout=0.5):
        """
        Read response from MCU
        
        Args:
            expected_id: Expected message ID (optional)
            timeout: Read timeout in seconds
            
        Returns:
            dict: Parsed response or None if failed
        """
        if not self.ser or not self.ser.is_open:
            return None
        
        try:
            # Set timeout for this read
            old_timeout = self.ser.timeout
            self.ser.timeout = timeout
            
            # Read until we get a complete message
            while True:
                # Look for start of frame
                sof = self.ser.read(1)
                if not sof or sof[0] != SOF:
                    continue
                
                # Read command/message ID
                msg_id = self.ser.read(1)
                if not msg_id:
                    continue
                msg_id = msg_id[0]
                
                # Read payload length
                payload_len = self.ser.read(1)
                if not payload_len:
                    continue
                payload_len = payload_len[0]
                
                # Read payload
                payload = self.ser.read(payload_len)
                if len(payload) != payload_len:
                    continue
                
                # Read checksum
                checksum = self.ser.read(1)
                if not checksum:
                    continue
                checksum = checksum[0]
                
                # Read end of frame
                eof = self.ser.read(1)
                if not eof or eof[0] != EOF:
                    continue
                
                # Verify checksum
                calculated_checksum = self._calculate_checksum(msg_id, payload)
                if checksum != calculated_checksum:
                    self.logger.warning(f"Checksum mismatch: expected {calculated_checksum:02X}, got {checksum:02X}")
                    continue
                
                # Check if this is the expected message
                if expected_id is not None and msg_id != expected_id:
                    self.logger.debug(f"Unexpected message ID: {msg_id:02X}, expected {expected_id:02X}")
                    continue
                
                # Restore timeout
                self.ser.timeout = old_timeout
                
                self.logger.debug(f"Received: {msg_id:02X} with {len(payload)} bytes payload")
                return {
                    'id': msg_id,
                    'payload': payload,
                    'checksum': checksum
                }
                
        except Exception as e:
            self.logger.error(f"Failed to read response: {e}")
            return None
        finally:
            # Restore timeout
            if self.ser:
                self.ser.timeout = old_timeout
    
    def set_motors(self, motor1_speed, motor2_speed):
        """
        Set motor speeds
        
        Args:
            motor1_speed: Speed for motor 1 (-1000 to 1000)
            motor2_speed: Speed for motor 2 (-1000 to 1000)
            
        Returns:
            bool: True if command sent successfully
        """
        # Clamp speeds to valid range
        motor1_speed = max(-1000, min(1000, int(motor1_speed)))
        motor2_speed = max(-1000, min(1000, int(motor2_speed)))
        
        # Pack speeds as little-endian signed 16-bit integers
        payload = struct.pack('<hh', motor1_speed, motor2_speed)
        
        success = self._send_message(CMD_SET_MOTORS, payload)
        if success:
            self.logger.info(f"Set motors: M1={motor1_speed}, M2={motor2_speed}")
        return success
    
    def get_encoders(self):
        """
        Get current encoder values
        
        Returns:
            tuple: (encoder1_value, encoder2_value) or (None, None) if failed
        """
        if not self._send_message(CMD_GET_ENCODERS):
            return None, None
        
        response = self._read_response(MSG_ENCODER_DATA, timeout=1.0)
        if not response or len(response['payload']) != 8:
            self.logger.error("Invalid encoder response")
            return None, None
        
        # Unpack as little-endian signed 32-bit integers
        encoder1, encoder2 = struct.unpack('<ii', response['payload'])
        self.logger.info(f"Encoders: E1={encoder1}, E2={encoder2}")
        return encoder1, encoder2
    
    def reset_encoders(self):
        """
        Reset encoder counts to zero
        
        Returns:
            bool: True if command sent successfully
        """
        success = self._send_message(CMD_RESET_ENCODERS)
        if success:
            self.logger.info("Reset encoders")
        return success
    
    def ping(self):
        """
        Ping the MCU to check if it's responsive
        
        Returns:
            bool: True if MCU responds with pong
        """
        if not self._send_message(CMD_PING):
            return False
        
        response = self._read_response(MSG_PONG, timeout=1.0)
        if response:
            self.logger.info("MCU is responsive")
            return True
        else:
            self.logger.warning("MCU did not respond to ping")
            return False
    
    def move_steps(self, motor1_steps, motor2_steps):
        """
        Move motors a specified number of steps
        
        Args:
            motor1_steps: Steps for motor 1 (signed 32-bit)
            motor2_steps: Steps for motor 2 (signed 32-bit)
            
        Returns:
            bool: True if command sent successfully
        """
        # Pack steps as little-endian signed 32-bit integers
        payload = struct.pack('<ii', int(motor1_steps), int(motor2_steps))
        
        success = self._send_message(CMD_MOVE_STEPS, payload)
        if success:
            self.logger.info(f"Move steps: M1={motor1_steps}, M2={motor2_steps}")
        return success
    
    def stop_motors(self):
        """Stop both motors (set speed to 0)"""
        return self.set_motors(0, 0)


def main():
    """Example usage"""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create motor controller
    motor = MotorController()
    
    try:
        # Connect to MCU
        if not motor.connect():
            print("Failed to connect to MCU")
            return
        
        # Test ping
        if not motor.ping():
            print("MCU is not responsive")
            return
        
        # Test motor control
        print("Testing motor control...")
        motor.set_motors(500, -500)  # Move forward
        print("Moving forward for 3 seconds...")
        time.sleep(3)
        
        motor.set_motors(-500, 500)  # Move backward
        print("Moving backward for 3 seconds...")
        time.sleep(3)
        
        motor.stop_motors()  # Stop
        print("Motors stopped")
        time.sleep(3)
        
        # Test encoder reading
        print("Testing encoder reading...")
        encoder1, encoder2 = motor.get_encoders()
        time.sleep(3)
        if encoder1 is not None:
            print(f"Encoder values: E1={encoder1}, E2={encoder2}")
        
        # Test step movement
        print("Testing step movement...")
        motor.move_steps(1000, -1000)
        time.sleep(3)
        
        # Reset encoders
        motor.reset_encoders()
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        motor.stop_motors()
        motor.disconnect()


if __name__ == "__main__":
    main()
