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
- Virtual environment support

## Quick Start

1. **Install system dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip libatlas-base-dev portaudio19-dev
   ```

2. **Install Python packages:**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Test microphone:**
   ```bash
   python3 -m sounddevice  # List available audio devices
   ```

4. **Test microphone:**
   ```bash
   arecord -l  # List available audio devices
   arecord -D hw:1,0 -f cd -t wav test.wav  # Test recording
   ```

5. **YAMNet model is included:**
   - The project includes a pre-trained YAMNet model in the `yamnet_model/` directory
   - YAMNet is a powerful audio classification model that can detect 521 different sound classes
   - The model will automatically detect fart sounds (class index 55)

6. **Run the detector:**
   ```bash
   python3 fart_detector.py
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
- **Fart Detection**: Specifically looks for "Fart" class (index 55)
- **Input**: 16kHz mono audio
- **Output**: Confidence scores for all 521 classes

## Troubleshooting

### Audio Issues
- Ensure USB microphone is properly connected
- Check audio device permissions
- Verify ALSA configuration

### Performance Issues
- Monitor CPU usage during detection
- Adjust sample rate if needed
- Consider using TensorFlow Lite for better performance

### Model Issues
- Ensure model file is in the correct format
- Check model input shape compatibility
- Verify feature extraction pipeline

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