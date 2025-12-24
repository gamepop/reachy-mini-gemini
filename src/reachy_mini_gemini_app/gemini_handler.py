"""Gemini Live API handler for real-time audio/video conversations.

This module handles bidirectional audio/video streaming with Google's Gemini Live API.
Supports both local PyAudio and Reachy Mini hardware for audio I/O.

Audio format: raw PCM, 16-bit little-endian
Send sample rate: 16kHz
Receive sample rate: 24kHz
"""

import asyncio
import base64
import io
import logging
import os
import struct
import threading
import time
import traceback
from typing import Optional

import cv2
import numpy as np
from scipy import signal

from google import genai
from google.genai import types

from reachy_mini import ReachyMini
from reachy_mini_gemini_app.movements import MovementController

logger = logging.getLogger(__name__)

# Audio configuration (defaults, can be overridden via CLI)
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHANNELS = 1
DEFAULT_CHUNK_SIZE = 512

# Gemini model for native audio
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"

# Try to import PyAudio (optional, for local audio)
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

SYSTEM_INSTRUCTION = """You are Reachy Mini, a small expressive robot made by Pollen Robotics.
You have a head that can move in all directions and two antennas on top that can express emotions.

Personality:
- You are friendly, curious, and playful
- You enjoy having conversations with humans
- You express yourself through head movements and antenna positions
- Keep responses concise since you're speaking them aloud

Available movement tools - use them frequently to be expressive:
- move_head: Look in a direction (left, right, up, down, center)
- move_head_precise: Fine control over head orientation with roll, pitch, yaw angles
- express_emotion: Express emotions (happy, sad, surprised, curious, excited, sleepy, confused, angry, love)
- move_antennas: Control antenna angles individually
- antenna_expression: Quick antenna presets (neutral, alert, droopy, asymmetric, perky)
- nod_yes: Nod your head yes
- shake_no: Shake your head no
- tilt_head: Tilt head to one side (curious look)
- look_at_camera: Look directly at the person
- do_dance: Dance! (default, happy, or silly style)
- wake_up: Wake up animation
- go_to_sleep: Sleep animation
- reset_position: Return to neutral pose

Be expressive! Move your head and antennas while talking to show engagement and emotion."""


class GeminiLiveHandler:
    """Handles real-time audio/video conversation with Gemini Live API."""

    def __init__(
        self,
        api_key: str,
        robot: ReachyMini,
        movement_controller: MovementController,
        use_camera: bool = True,
        use_robot_audio: bool = False,
        # Audio settings
        mic_gain: float = 3.0,
        chunk_size: int = 512,
        send_queue_size: int = 5,
        recv_queue_size: int = 8,
        # Video settings
        camera_fps: float = 1.0,
        jpeg_quality: int = 50,
        camera_width: int = 640,
    ):
        """Initialize the Gemini Live handler.

        Args:
            api_key: Google API key
            robot: ReachyMini instance
            movement_controller: Controller for robot movements
            use_camera: Whether to enable camera/vision capabilities
            use_robot_audio: Whether to use Reachy Mini's mic/speaker instead of local
            mic_gain: Microphone gain multiplier
            chunk_size: Audio chunk size in samples
            send_queue_size: Output queue size
            recv_queue_size: Input queue size
            camera_fps: Frames per second for camera
            jpeg_quality: JPEG compression quality
            camera_width: Max camera frame width
        """
        self.robot = robot
        self.movement_controller = movement_controller
        self.use_camera = use_camera
        self.use_robot_audio = use_robot_audio

        # Configurable audio settings
        self.mic_gain = mic_gain
        self.chunk_size = chunk_size
        self.send_queue_size = send_queue_size
        self.recv_queue_size = recv_queue_size

        # Configurable video settings
        self.camera_fps = camera_fps
        self.jpeg_quality = jpeg_quality
        self.camera_width = camera_width

        # Log configuration
        logger.info(f"Audio config: mic_gain={mic_gain}, chunk_size={chunk_size}")
        logger.info(f"Queue config: send={send_queue_size}, recv={recv_queue_size}")
        logger.info(f"Video config: fps={camera_fps}, quality={jpeg_quality}, width={camera_width}")

        # Initialize Gemini client with v1beta API
        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )

        # Audio setup (PyAudio for local, or robot hardware)
        self.pya = None
        self.audio_stream: Optional[object] = None

        if not use_robot_audio and PYAUDIO_AVAILABLE:
            self.pya = pyaudio.PyAudio()
        elif not use_robot_audio and not PYAUDIO_AVAILABLE:
            logger.warning("PyAudio not available, falling back to robot audio")
            self.use_robot_audio = True

        # Session and queues
        self.session = None
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None

        # Camera frame rate control
        self.last_frame_time = 0

        # Define tools for the model
        self.tools = self._create_tools()

    def _create_tools(self) -> list:
        """Create function tools for the model."""
        move_head_tool = types.FunctionDeclaration(
            name="move_head",
            description="Move Reachy Mini's head to look in a direction",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "direction": types.Schema(
                        type=types.Type.STRING,
                        enum=["left", "right", "up", "down", "center"],
                        description="Direction to look",
                    ),
                },
                required=["direction"],
            ),
        )

        move_head_precise_tool = types.FunctionDeclaration(
            name="move_head_precise",
            description="Move head to precise orientation angles. Roll tilts head sideways, pitch looks up/down, yaw turns left/right.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "roll": types.Schema(
                        type=types.Type.NUMBER,
                        description="Roll angle in degrees (-30 to 30). Positive tilts right.",
                    ),
                    "pitch": types.Schema(
                        type=types.Type.NUMBER,
                        description="Pitch angle in degrees (-30 to 30). Positive looks down.",
                    ),
                    "yaw": types.Schema(
                        type=types.Type.NUMBER,
                        description="Yaw angle in degrees (-45 to 45). Positive turns right.",
                    ),
                },
            ),
        )

        express_emotion_tool = types.FunctionDeclaration(
            name="express_emotion",
            description="Express an emotion through head movement and antennas",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "emotion": types.Schema(
                        type=types.Type.STRING,
                        enum=["happy", "sad", "surprised", "curious", "excited", "sleepy", "confused", "angry", "love"],
                        description="Emotion to express",
                    ),
                },
                required=["emotion"],
            ),
        )

        move_antennas_tool = types.FunctionDeclaration(
            name="move_antennas",
            description="Move antennas to specific angles",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "right_angle": types.Schema(
                        type=types.Type.NUMBER,
                        description="Right antenna angle in degrees (-90 to 90)",
                    ),
                    "left_angle": types.Schema(
                        type=types.Type.NUMBER,
                        description="Left antenna angle in degrees (-90 to 90)",
                    ),
                },
            ),
        )

        antenna_expression_tool = types.FunctionDeclaration(
            name="antenna_expression",
            description="Set antennas to a preset expression",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        enum=["neutral", "alert", "droopy", "asymmetric", "perky"],
                        description="Antenna expression preset",
                    ),
                },
                required=["expression"],
            ),
        )

        nod_yes_tool = types.FunctionDeclaration(
            name="nod_yes",
            description="Nod head up and down to indicate yes or agreement",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "times": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of nods (1-5, default 2)",
                    ),
                },
            ),
        )

        shake_no_tool = types.FunctionDeclaration(
            name="shake_no",
            description="Shake head left and right to indicate no or disagreement",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "times": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of shakes (1-5, default 2)",
                    ),
                },
            ),
        )

        tilt_head_tool = types.FunctionDeclaration(
            name="tilt_head",
            description="Tilt head to one side, like a curious dog",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "direction": types.Schema(
                        type=types.Type.STRING,
                        enum=["left", "right"],
                        description="Direction to tilt",
                    ),
                    "angle": types.Schema(
                        type=types.Type.NUMBER,
                        description="Tilt angle in degrees (5-30, default 20)",
                    ),
                },
                required=["direction"],
            ),
        )

        look_at_camera_tool = types.FunctionDeclaration(
            name="look_at_camera",
            description="Look directly at the camera/person",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        )

        do_dance_tool = types.FunctionDeclaration(
            name="do_dance",
            description="Perform a fun dance animation",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "style": types.Schema(
                        type=types.Type.STRING,
                        enum=["default", "happy", "silly"],
                        description="Dance style (default: default)",
                    ),
                },
            ),
        )

        wake_up_tool = types.FunctionDeclaration(
            name="wake_up",
            description="Perform wake up animation - use when greeting someone or starting a conversation",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        )

        go_to_sleep_tool = types.FunctionDeclaration(
            name="go_to_sleep",
            description="Perform sleep animation - use when saying goodbye or ending conversation",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        )

        reset_position_tool = types.FunctionDeclaration(
            name="reset_position",
            description="Reset head and antennas to neutral position",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        )

        all_tools = [
            move_head_tool,
            move_head_precise_tool,
            express_emotion_tool,
            move_antennas_tool,
            antenna_expression_tool,
            nod_yes_tool,
            shake_no_tool,
            tilt_head_tool,
            look_at_camera_tool,
            do_dance_tool,
            wake_up_tool,
            go_to_sleep_tool,
            reset_position_tool,
        ]

        return [types.Tool(function_declarations=all_tools)]

    async def _handle_tool_call(self, tool_call) -> str:
        """Handle a function call from the model."""
        name = tool_call.name
        args = dict(tool_call.args) if tool_call.args else {}

        logger.info(f"Tool call: {name}({args})")

        try:
            if name == "move_head":
                direction = args.get("direction", "center")
                return await self.movement_controller.move_head(direction)

            elif name == "move_head_precise":
                roll = args.get("roll", 0)
                pitch = args.get("pitch", 0)
                yaw = args.get("yaw", 0)
                return await self.movement_controller.move_head_precise(roll, pitch, yaw)

            elif name == "express_emotion":
                emotion = args.get("emotion", "happy")
                return await self.movement_controller.express_emotion(emotion)

            elif name == "move_antennas":
                right_angle = args.get("right_angle", 0)
                left_angle = args.get("left_angle", 0)
                return await self.movement_controller.move_antennas(right_angle, left_angle)

            elif name == "antenna_expression":
                expression = args.get("expression", "neutral")
                return await self.movement_controller.antenna_expression(expression)

            elif name == "nod_yes":
                times = args.get("times", 2)
                return await self.movement_controller.nod_yes(times)

            elif name == "shake_no":
                times = args.get("times", 2)
                return await self.movement_controller.shake_no(times)

            elif name == "tilt_head":
                direction = args.get("direction", "left")
                angle = args.get("angle", 20)
                return await self.movement_controller.tilt_head(direction, angle)

            elif name == "look_at_camera":
                return await self.movement_controller.look_at_camera()

            elif name == "do_dance":
                style = args.get("style", "default")
                return await self.movement_controller.do_dance(style)

            elif name == "wake_up":
                return await self.movement_controller.wake_up()

            elif name == "go_to_sleep":
                return await self.movement_controller.go_to_sleep()

            elif name == "reset_position":
                return await self.movement_controller.reset_position()

            else:
                logger.warning(f"Unknown tool: {name}")
                return f"Unknown tool: {name}"

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error: {e}"

    async def listen_audio(self) -> None:
        """Continuously capture audio from microphone and queue it."""
        if self.use_robot_audio:
            await self._listen_audio_robot()
        else:
            await self._listen_audio_local()

    async def _listen_audio_local(self) -> None:
        """Capture audio from local microphone using PyAudio."""
        # Initialize PyAudio if not already done
        if self.pya is None:
            if not PYAUDIO_AVAILABLE:
                logger.error("PyAudio not available and robot audio not working")
                return
            self.pya = pyaudio.PyAudio()

        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=self.chunk_size,
        )

        kwargs = {"exception_on_overflow": False}

        while True:
            data = await asyncio.to_thread(self.audio_stream.read, self.chunk_size, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def _listen_audio_robot(self) -> None:
        """Capture audio from Reachy Mini's microphone."""
        # Check if audio is actually available
        if not hasattr(self.robot.media, 'audio') or self.robot.media.audio is None:
            logger.warning("Robot audio not available, falling back to local audio")
            self.use_robot_audio = False
            await self._listen_audio_local()
            return

        logger.info("Starting Reachy Mini microphone recording...")
        await asyncio.to_thread(self.robot.media.start_recording)

        # Audio gain from config
        mic_gain = self.mic_gain

        while True:
            try:
                # Get audio sample from robot (bytes or numpy array)
                sample = await asyncio.to_thread(self.robot.media.get_audio_sample)

                if sample is None:
                    await asyncio.sleep(0.005)
                    continue

                # Convert to bytes if numpy array
                if isinstance(sample, np.ndarray):
                    # Robot sends stereo float32, convert to mono
                    if sample.dtype == np.float32:
                        # If stereo (shape is (N, 2)), convert to mono
                        if len(sample.shape) == 2 and sample.shape[1] == 2:
                            sample = np.mean(sample, axis=1)

                        # Apply gain and clip to prevent distortion
                        sample = sample * mic_gain
                        sample = np.clip(sample, -1.0, 1.0)

                        # Convert to int16 PCM
                        sample = (sample * 32767).astype(np.int16)
                    data = sample.tobytes()
                else:
                    data = sample

                if data:
                    try:
                        self.out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
                    except asyncio.QueueFull:
                        # Drop old audio to keep stream fresh
                        try:
                            self.out_queue.get_nowait()
                            self.out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
                        except Exception:
                            pass

            except Exception as e:
                logger.debug(f"Audio capture error: {e}")
                await asyncio.sleep(0.01)

    async def send_realtime(self) -> None:
        """Send queued audio/data to Gemini."""
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def receive_audio(self) -> None:
        """Receive responses from Gemini and handle them."""
        while True:
            try:
                turn = self.session.receive()
                async for response in turn:
                    # Handle audio data
                    if data := response.data:
                        try:
                            self.audio_in_queue.put_nowait(data)
                        except asyncio.QueueFull:
                            # Drop oldest audio to make room
                            try:
                                self.audio_in_queue.get_nowait()
                                self.audio_in_queue.put_nowait(data)
                            except Exception:
                                pass
                        continue

                    # Handle text (print transcription)
                    if hasattr(response, 'text') and response.text:
                        print(response.text, end="", flush=True)

                    # Handle tool calls
                    if hasattr(response, 'tool_call') and response.tool_call:
                        for fc in response.tool_call.function_calls:
                            logger.debug(f"Processing tool call: {fc.name}")
                            result = await self._handle_tool_call(fc)
                            logger.debug(f"Tool result: {result}")

                            # Send tool response back to Gemini
                            try:
                                await self.session.send(
                                    input=types.LiveClientToolResponse(
                                        function_responses=[
                                            types.FunctionResponse(
                                                name=fc.name,
                                                id=fc.id,
                                                response={"result": result},
                                            )
                                        ]
                                    )
                                )
                            except Exception as e:
                                logger.error(f"Failed to send tool response: {e}")

                # Only clear queue if user interrupted (queue is full)
                # This prevents cutting off normal responses
                if self.audio_in_queue.qsize() > self.recv_queue_size - 2:
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()

            except Exception as e:
                logger.warning(f"Receive audio error: {e}")
                await asyncio.sleep(0.1)

    async def play_audio(self) -> None:
        """Play received audio from queue."""
        if self.use_robot_audio:
            await self._play_audio_robot()
        else:
            await self._play_audio_local()

    async def _play_audio_local(self) -> None:
        """Play audio through local speakers using PyAudio."""
        # Initialize PyAudio if not already done
        if self.pya is None:
            if not PYAUDIO_AVAILABLE:
                logger.error("PyAudio not available and robot audio not working")
                # Just drain the queue to avoid blocking
                while True:
                    await self.audio_in_queue.get()
                return
            self.pya = pyaudio.PyAudio()

        stream = await asyncio.to_thread(
            self.pya.open,
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def _play_audio_robot(self) -> None:
        """Play audio through Reachy Mini's speaker."""
        # Check if audio is actually available
        if not hasattr(self.robot.media, 'audio') or self.robot.media.audio is None:
            logger.warning("Robot speaker not available, falling back to local audio")
            self.use_robot_audio = False
            await self._play_audio_local()
            return

        logger.info("Starting Reachy Mini speaker playback...")
        await asyncio.to_thread(self.robot.media.start_playing)

        # Robot expects 16kHz, Gemini sends 24kHz
        ROBOT_SAMPLE_RATE = 16000

        while True:
            try:
                bytestream = await self.audio_in_queue.get()

                # Convert bytes to numpy float32 array for robot speaker
                # Input is 16-bit PCM at 24kHz
                audio_int16 = np.frombuffer(bytestream, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32767.0

                # Resample from 24kHz to 16kHz
                num_samples = int(len(audio_float32) * ROBOT_SAMPLE_RATE / RECEIVE_SAMPLE_RATE)
                audio_resampled = signal.resample(audio_float32, num_samples)

                await asyncio.to_thread(self.robot.media.push_audio_sample, audio_resampled.astype(np.float32))

            except Exception as e:
                logger.debug(f"Audio playback error: {e}")
                await asyncio.sleep(0.01)

    async def stream_camera(self) -> None:
        """Stream camera frames to Gemini."""
        if not self.use_camera:
            # Keep task alive but do nothing
            while True:
                await asyncio.sleep(10)
            return

        # Check if camera is actually available
        if not hasattr(self.robot.media, 'camera') or self.robot.media.camera is None:
            logger.warning("Robot camera not available, disabling camera streaming")
            self.use_camera = False
            # Keep task alive
            while True:
                await asyncio.sleep(10)
            return

        logger.info("Starting camera streaming...")

        # Wait for WebRTC to establish before checking camera
        await asyncio.sleep(3.0)

        # Track consecutive failures to avoid spam
        consecutive_failures = 0
        max_failures = 30  # Give more time for WebRTC camera stream

        while True:
            try:
                current_time = time.time()

                # Rate limit camera frames
                if current_time - self.last_frame_time < (1.0 / self.camera_fps):
                    await asyncio.sleep(0.05)
                    continue

                # Get frame from robot camera
                frame = await asyncio.to_thread(self.robot.media.get_frame)

                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        if consecutive_failures == max_failures:
                            logger.warning("Camera not responding, will keep retrying...")
                    await asyncio.sleep(0.5)
                    continue

                consecutive_failures = 0
                self.last_frame_time = current_time

                # Resize frame for efficiency
                h, w = frame.shape[:2]
                if w > self.camera_width:
                    scale = self.camera_width / w
                    frame = cv2.resize(frame, (self.camera_width, int(h * scale)))

                # Encode as JPEG with configurable quality
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                image_bytes = buffer.tobytes()

                # Send to Gemini (non-blocking, skip if queue full)
                try:
                    self.out_queue.put_nowait({
                        "data": image_bytes,
                        "mime_type": "image/jpeg"
                    })
                    logger.debug(f"Sent camera frame ({len(image_bytes)} bytes)")
                except asyncio.QueueFull:
                    logger.debug("Skipping camera frame, queue full")

            except Exception as e:
                logger.debug(f"Camera streaming error: {e}")
                consecutive_failures += 1
                await asyncio.sleep(0.5)

    async def run(self, stop_event: threading.Event) -> None:
        """Run the conversation loop with auto-reconnection.

        Args:
            stop_event: Event to signal when to stop
        """
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=SYSTEM_INSTRUCTION)]
            ),
            tools=self.tools,
        )

        while not stop_event.is_set():
            try:
                async with (
                    self.client.aio.live.connect(model=MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    self.audio_in_queue = asyncio.Queue(maxsize=self.recv_queue_size)
                    self.out_queue = asyncio.Queue(maxsize=self.send_queue_size)

                    audio_source = "robot" if self.use_robot_audio else "local"
                    camera_status = "enabled" if self.use_camera else "disabled"
                    logger.info(f"Connected to Gemini Live API (audio: {audio_source}, camera: {camera_status})")
                    print(f"\n🎤 Speak to Reachy Mini! (audio: {audio_source}, camera: {camera_status})")
                    print("Press Ctrl+C to stop.\n")

                    # Start all tasks
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())

                    # Start camera streaming if enabled
                    if self.use_camera:
                        tg.create_task(self.stream_camera())

                    # Wait for stop signal
                    while not stop_event.is_set():
                        await asyncio.sleep(0.1)

                    raise asyncio.CancelledError("Stop requested")

            except asyncio.CancelledError:
                break
            except ExceptionGroup as EG:
                await self._cleanup_streams()
                # Log what caused the exception group
                for exc in EG.exceptions:
                    logger.warning(f"Task exception: {type(exc).__name__}: {exc}")
                    logger.debug(traceback.format_exception(exc))
                # Check if it's a connection error - reconnect
                logger.warning("Connection lost, reconnecting in 2 seconds...")
                print("\n⚠️ Connection lost. Reconnecting...\n")
                await asyncio.sleep(2)
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"\n❌ Error: {e}. Reconnecting in 2 seconds...\n")
                await asyncio.sleep(2)
                continue

    async def _cleanup_streams(self) -> None:
        """Clean up audio streams."""
        if self.audio_stream:
            try:
                if hasattr(self.audio_stream, 'close'):
                    self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None

        if self.use_robot_audio and self.robot and self.robot.media:
            try:
                await asyncio.to_thread(self.robot.media.stop_recording)
            except Exception:
                pass
            try:
                await asyncio.to_thread(self.robot.media.stop_playing)
            except Exception:
                pass

    async def close(self) -> None:
        """Clean up resources."""
        await self._cleanup_streams()

        if self.pya:
            self.pya.terminate()
            self.pya = None

        logger.info("Gemini handler closed")
