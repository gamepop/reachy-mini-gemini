# Reachy Mini Gemini Live App

Talk with Reachy Mini using Google's Gemini Live API for real-time voice conversations.

## Features

- Real-time voice conversation via Gemini Live API
- **Camera streaming**: Send robot's camera feed to Gemini for vision capabilities
- **Robot audio**: Use Reachy Mini's microphone and speaker (or local audio)
- Expressive head movements and antenna animations
- Function calling for robot control (move head, express emotions)
- Support for both wired and wireless Reachy Mini
- **Configurable audio/video settings** for tuning latency and quality

## Requirements

- Python 3.10+
- Reachy Mini robot (or simulation)
- Google API key with Gemini access
- PortAudio (for PyAudio)

## Installation

```bash
# Install PortAudio (macOS)
brew install portaudio

# Install PortAudio (Ubuntu)
sudo apt-get install portaudio19-dev python3-dev

# Install the app (basic - local audio only)
pip install -e .

# Install with wireless/GStreamer support (for robot audio/camera over network)
pip install -e ".[wireless]"
```

## Configuration

1. Copy `.env.example` to `.env`
2. Add your Google API key:
   ```
   GOOGLE_API_KEY=your-key-here
   ```

## Usage

First, make sure the Reachy Mini daemon is running:
```bash
# For simulation
reachy-mini-daemon --sim

# For real hardware (wired)
reachy-mini-daemon

# For wireless Reachy Mini (on the robot)
reachy-mini-daemon --wireless-version
```

Then run the app:
```bash
# Basic usage (local mic/speaker, robot camera enabled)
reachy-mini-gemini

# Wireless Reachy Mini
reachy-mini-gemini --wireless

# Use robot's microphone and speaker (instead of local audio)
reachy-mini-gemini --robot-audio

# Full robot mode (robot camera + robot audio)
reachy-mini-gemini --wireless --robot-audio

# Without camera
reachy-mini-gemini --no-camera

# Debug mode
reachy-mini-gemini --debug
```

## Command Line Options

### Basic Options

| Option | Description |
|--------|-------------|
| `--wireless` | Connect to wireless Reachy Mini |
| `--robot-audio` | Use Reachy Mini's microphone and speaker |
| `--no-camera` | Disable camera streaming to Gemini |
| `--debug` | Enable debug logging |

### Audio Settings

| Option | Default | Description |
|--------|---------|-------------|
| `--mic-gain` | 3.0 | Microphone gain multiplier (1.0-10.0). Increase if Gemini can't hear you |
| `--chunk-size` | 512 | Audio chunk size in samples (256-2048). Smaller = lower latency |
| `--send-queue-size` | 5 | Output queue size for sending audio/video (1-20) |
| `--recv-queue-size` | 8 | Input queue size for receiving audio (1-20) |

### Video Settings

| Option | Default | Description |
|--------|---------|-------------|
| `--camera-fps` | 1.0 | Camera frames per second to send (0.5-5.0) |
| `--jpeg-quality` | 50 | JPEG compression quality (10-95) |
| `--camera-width` | 640 | Max camera frame width (320-1280) |

### Tuning Examples

```bash
# If Gemini can't hear well - boost mic gain
reachy-mini-gemini --wireless --robot-audio --mic-gain 5.0

# Lower latency (smaller buffers, may cause choppy audio)
reachy-mini-gemini --wireless --robot-audio --send-queue-size 3 --recv-queue-size 5 --chunk-size 256

# Better video quality (higher bandwidth)
reachy-mini-gemini --wireless --robot-audio --jpeg-quality 70 --camera-fps 2.0

# Reduce bandwidth (lower quality, good for slow networks)
reachy-mini-gemini --wireless --robot-audio --jpeg-quality 30 --camera-fps 0.5 --camera-width 320

# Disable camera if connection is unstable
reachy-mini-gemini --wireless --robot-audio --no-camera
```

## Wireless Media Setup (Advanced)

To use the robot's camera, microphone, and speaker over wireless, you need to build and install the GStreamer WebRTC Rust plugins. This is required for full wireless functionality.

### Prerequisites

```bash
# Ubuntu/Debian - Install build dependencies
sudo apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libnice-dev \
    libssl-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad
```

### Install Rust (if not already installed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Build GStreamer WebRTC Rust Plugin

```bash
# Clone gst-plugins-rs
cd /tmp
git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git
cd gst-plugins-rs

# Checkout version matching your GStreamer (check with: gst-inspect-1.0 --version)
# For GStreamer 1.24.x:
git checkout gstreamer-1.24.13

# Build the webrtc plugin (takes a few minutes)
cargo build --release -p gst-plugin-webrtc

# Install the plugin
sudo cp target/release/libgstrswebrtc.so /usr/lib/$(uname -m)-linux-gnu/gstreamer-1.0/

# Verify installation
gst-inspect-1.0 webrtcsrc
```

### Install Python GStreamer bindings

```bash
# Install reachy_mini with gstreamer support
pip install "reachy_mini[gstreamer]"
```

### Verify Setup

```bash
# Should show "WebRTCSrc" details
gst-inspect-1.0 webrtcsrc

# Test the app
reachy-mini-gemini --wireless --robot-audio --debug
```

### Troubleshooting Wireless

| Issue | Solution |
|-------|----------|
| "No module named 'gi'" | Install `python3-gi` or symlink system gi to venv |
| "Failed to create webrtcsrc element" | Build and install gst-plugins-rs (see above) |
| "Robot audio not available" | Check that `reachy_mini[gstreamer]` is installed |
| Connection keeps dropping | Try `--no-camera` or reduce `--camera-fps` |
| Audio too quiet | Increase `--mic-gain` to 4.0 or 5.0 |

## Available Tools

The Gemini model can use these tools during conversation:

| Tool | Description |
|------|-------------|
| `move_head` | Look left, right, up, down, or center |
| `express_emotion` | Express happy, sad, surprised, curious, excited, or sleepy |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Reachy Mini                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────────────┐   │
│  │ Camera  │  │   Mic   │  │ Speaker │  │ Head + Antennas│   │
│  └────┬────┘  └────┬────┘  └────▲────┘  └───────▲────────┘   │
└───────┼────────────┼────────────┼───────────────┼────────────┘
        │            │            │               │
        │ (JPEG)     │ (PCM 16k)  │ (PCM 24k)     │
        ▼            ▼            │               │
┌───────────────────────────────────────────────────────────┐
│                    Gemini Live API                        │
│  - Real-time audio conversation                           │
│  - Vision understanding (camera frames)                   │
│  - Function calling (robot control)                       │
└────────────────────────┬──────────────────────────────────┘
                         │
                ┌────────▼─────────┐
                │   Tool Calls     │
                │  move_head       │
                │  express_emotion │
                └──────────────────┘
```

### Audio Modes

- **Local audio (default)**: Uses your computer's microphone and speakers
- **Robot audio (`--robot-audio`)**: Uses Reachy Mini's built-in mic and speaker via WebRTC

### Audio Processing

- Input: 16kHz mono PCM (resampled from robot's stereo)
- Output: 24kHz mono PCM (resampled to robot's 16kHz)
- Configurable gain for microphone boost

## License

Apache 2.0
