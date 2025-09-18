# Simple Raspberry Pi Fart Detector using YAMNet

A simple audio classification system that runs on Raspberry Pi 4 to detect fart sounds using the YAMNet pre-trained model. The system continuously listens to a USB microphone, processes audio through YAMNet, and prints detection results.

## Features

- Simple continuous audio capture from USB microphone
- Machine learning-based sound classification
- Configurable detection thresholds
- Easy setup and deployment
- Optimized for Raspberry Pi 4 performance

## Hardware Requirements

- Raspberry Pi 4 (4GB RAM recommended)
- USB Microphone (compatible with ALSA)
- MicroSD card (32GB+ recommended)
- Power supply (5V, 3A)

## Software Requirements

- Raspberry Pi OS (64-bit recommended)
- Python 3.8+
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
   sudo apt install python3-pip tensorflow portaudio19-dev
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

5. **Test microphone:**
   ```bash
   python3 -m sounddevice  # List available audio devices
   arecord -l  # List available audio devices
   arecord -D hw:1,0 -f cd -t wav test.wav  # Test recording
   ```

6. **YAMNet model is included:**
   - The project includes a pre-trained YAMNet model in the `yamnet_model/` directory
   - YAMNet is a powerful audio classification model that can detect 521 different sound classes
   - The model will automatically detect fart sounds (class index 55)

7. **Run the detector:**
   ```bash
   source .venv/bin/activate
   python fart_detector.py
   ```

## Configuration

Edit the global constants in `fart_detector.py` to adjust:
- `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
- `DURATION`: Recording duration in seconds (default: 2.0)
- `DEVICE_ID`: Audio device ID (default: None for auto-detect)
- `DETECTION_THRESHOLD`: Confidence threshold for detection (default: 0.35)

## Project Structure

```
geekcon-rpi-fart/
├── README.md
├── requirements.txt
├── fart_detector.py
└── yamnet_model/
    ├── saved_model.pb
    ├── yamnet_class_map.csv
    └── variables/
```

## YAMNet Model

The project uses the pre-trained YAMNet model:
- **YAMNet**: A powerful audio classification model from Google
- **Classes**: Can detect 521 different sound classes
- **Fart Detection**: Looks for "Fart" class (index 55) with case-insensitive matching
- **Input**: 16kHz mono audio
- **Output**: Confidence scores for all 521 classes

## Troubleshooting

### Installation Issues
- **TensorFlow Installation**: Use `tensorflow-aarch64` for ARM64 Raspberry Pi
- **NumPy Compatibility**: Pin to NumPy 1.x to avoid TensorFlow conflicts
- **Dependency Conflicts**: Use `--no-deps` flag with uv pip install
- **Virtual Environment**: Always activate `.venv` before running

### Audio Issues
- Ensure USB microphone is properly connected
- Check audio device permissions: `sudo usermod -a -G audio $USER`
- Verify ALSA configuration: `arecord -l`
- Test microphone: `arecord -D hw:1,0 -f cd -t wav test.wav`

### Performance Issues
- Monitor CPU usage during detection
- Adjust sample rate if needed (16kHz is optimal for YAMNet)
- Ensure adequate cooling for Raspberry Pi 4
- Close unnecessary applications to free up resources

### YAMNet Model Issues
- Verify model files are in `yamnet_model/` directory
- Check CSV file path: `yamnet_model/yamnet_class_map.csv`
- Ensure model loads without errors
- Test with known audio samples first

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
- Uses TensorFlow for machine learning