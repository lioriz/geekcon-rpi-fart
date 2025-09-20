# Raspberry Pi Fart Detection & Robot System

A comprehensive Raspberry Pi system with two main components:

1. **Fart Detector** (`fart_detector.py`) - A machine learning-based system that listens for fart sounds using TensorFlow Lite and YAMNet
2. **Farter Robot** (`main_farter.py`) - An autonomous robot that drives around randomly and plays fart sounds

Both systems are optimized for Raspberry Pi 4 and can be used independently or together.

## Features

### Fart Detector
- **Machine Learning**: Uses TensorFlow Lite with YAMNet for sound classification
- **Real-time Detection**: Continuously listens and detects fart sounds
- **Direction Finding**: Uses stereo microphones to determine sound direction
- **Motor Response**: Automatically drives toward detected fart sounds
- **Recording**: Optional audio recording and playback

### Farter Robot
- **Random Movement**: Robot drives with random left and right motor speeds (-500 to 500)
- **Random Sound Playback**: Plays fart sounds from MP3 files at random intervals
- **Multi-threaded**: Separate threads for movement and sound playback
- **Configurable**: Easy to adjust timing, speeds, and behavior
- **Motor Control**: Communicates with STM32L412xx MCU over UART

## Hardware Requirements

### For Fart Detector
- Raspberry Pi 4 (4GB RAM recommended)
- USB Microphone (compatible with ALSA) - stereo recommended for direction finding
- Motor controller (STM32L412xx MCU) connected via UART (optional)
- MicroSD card (32GB+ recommended)
- Power supply (5V, 3A)

### For Farter Robot
- Raspberry Pi 4 (4GB RAM recommended)
- Motor controller (STM32L412xx MCU) connected via UART
- Audio output (speakers or headphones)
- MicroSD card (32GB+ recommended)
- Power supply (5V, 3A)

## Software Requirements

- Raspberry Pi OS (64-bit recommended)
- Python 3.8+
- mpg123 for audio playback (Farter Robot only)
- uv (Python package manager)
- Virtual environment support

## Dependencies

This project uses specific package versions optimized for Raspberry Pi 4:

- **TensorFlow**: `tensorflow-aarch64==2.16.1` (ARM64 build)
- **NumPy**: `numpy==1.26.4` (pinned to 1.x for TensorFlow compatibility)
- **SoundDevice**: `sounddevice==0.5.2` (audio input)
- **Core Dependencies**: Pinned versions for stability

The `--no-deps` flag is used to avoid dependency conflicts on Raspberry Pi.

## Quick Start

1. **Install system dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip mpg123 portaudio19-dev
   ```

2. **Install uv (Python package manager):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create and activate virtual environment:**
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

4. **Install Python packages:**
   ```bash
   uv pip install --no-deps -r requirements.txt
   ```

### Option 1: Fart Detector
5. **Test microphone:**
   ```bash
   python3 -m sounddevice  # List available audio devices
   arecord -l  # List available audio devices
   ```

6. **Run the fart detector:**
   ```bash
   source .venv/bin/activate
   python3 fart_detector.py
   ```

### Option 2: Farter Robot
5. **Create fart sounds directory and add MP3 files:**
   ```bash
   mkdir -p ~/farts
   # Add your fart sound MP3 files to ~/farts/
   ```

6. **Run the farter robot:**
   ```bash
   source .venv/bin/activate
   python3 main_farter.py
   ```

### Stop either system:
Press `Ctrl+C` to stop gracefully.

## Configuration

### Fart Detector Configuration
Edit the global constants in `fart_detector.py`:
```python
# Audio settings
DEVICE_SAMPLE_RATE = 48000  # Audio sample rate
YAMNET_SAMPLE_RATE = 16000  # YAMNet model sample rate
DURATION = 1.0  # Recording duration in seconds
DETECTION_THRESHOLD = 0.003  # Detection sensitivity (lower = more sensitive)

# Motor settings (optional)
MOTOR_ENABLED = True  # Enable motor response
MOTOR_DRIVE_STEPS = 10  # Steps to drive after detection
```

### Farter Robot Configuration
Edit the configuration variables in `main_farter.py`:
```python
# Sound settings
MP3_DIR = os.path.expanduser("~/farts/*.mp3")  # Directory with MP3 files
VOLUME = 10  # Playback volume (0–32768)
SLEEP_BETWEEN_SOUNDS = (2, 8)  # Random interval between sounds (seconds)

# Movement settings
MOTOR_SPEED_RANGE = (-500, 500)  # Random motor speed range (min, max)
MOTOR_DURATION = (1, 5)  # Random movement duration (seconds)
SLEEP_BETWEEN_MOVEMENTS = (1, 5)  # Random interval between movements (seconds)
```

## How It Works

### Fart Detector
The detector uses machine learning to identify fart sounds:
1. **Audio Capture**: Continuously records from stereo microphones
2. **Direction Finding**: Uses time-difference-of-arrival (TDOA) to determine sound direction
3. **Sound Classification**: YAMNet model analyzes audio for fart sounds
4. **Motor Response**: Robot drives toward detected fart sounds
5. **Recording**: Optionally saves audio recordings for analysis

### Farter Robot Movement
The robot performs simple random movements:
- **Left Motor Speed**: Random value between -500 and 500
- **Right Motor Speed**: Random value between -500 and 500  
- **Movement Duration**: Random time between 1 and 5 seconds
- **Movement Interval**: Random pause between 1 and 5 seconds

This creates unpredictable driving patterns including:
- Forward/backward movement
- Left/right turns
- Spinning in place
- Curved paths

## Project Structure

```
geekcon-rpi-fart/
├── README.md
├── requirements.txt
├── main_farter.py          # Main farter robot program
├── fart_detector.py        # Original fart detection system
├── motor_control.py        # Motor controller interface
├── logger_utils.py         # Logging utilities
└── yamnet_model/           # YAMNet model files
    ├── saved_model.pb
    ├── yamnet_class_map.csv
    └── variables/
```

## Safety Notes

- The robot moves randomly and may bump into objects
- Ensure adequate space for movement
- Supervise operation, especially around pets or children
- Stop the robot immediately if it behaves unexpectedly

## Troubleshooting

### Fart Detector Issues
- **No audio input**: Check microphone connection and permissions
- **Low detection accuracy**: Adjust `DETECTION_THRESHOLD` in configuration
- **Motor not responding**: Check UART connection and motor controller power
- **Audio device errors**: Run `arecord -l` to list available devices

### Farter Robot Issues
- **No MP3 files found**: Ensure MP3 files are in `~/farts/` directory
- **Audio not playing**: Install mpg123: `sudo apt install mpg123`
- **Motor controller not responding**: Check UART connection (`/dev/serial0`)

### General Issues
- **Permission denied on serial port**: Add user to dialout group: `sudo usermod -a -G dialout $USER`
- **Audio device permissions**: Check audio group membership: `sudo usermod -a -G audio $USER`
- **Dependency conflicts**: Use `--no-deps` flag with uv pip install

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built for GeekCon 2024
- Optimized for Raspberry Pi 4
- Uses motor control and audio playback for autonomous robot behavior