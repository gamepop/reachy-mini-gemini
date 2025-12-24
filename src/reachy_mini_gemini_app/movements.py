"""Movement controller for Reachy Mini expressions and head movements.

This module provides high-level movement commands that the Gemini model
can use as tools during conversation.
"""

import asyncio
import logging
from typing import Optional, Tuple

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)


class MovementController:
    """Controls Reachy Mini head movements and expressions."""

    def __init__(self, robot: ReachyMini):
        """Initialize the movement controller.

        Args:
            robot: ReachyMini instance to control
        """
        self.robot = robot

    async def _goto_target(
        self,
        head: Optional[np.ndarray] = None,
        antennas: Optional[list] = None,
        duration: float = 0.5,
    ) -> None:
        """Wrapper for robot.goto_target that runs in executor (non-blocking async)."""
        try:
            await asyncio.to_thread(
                self.robot.goto_target,
                head=head,
                antennas=antennas,
                duration=duration,
            )
        except Exception as e:
            logger.error(f"Motion error: {e}")

    async def move_head(
        self,
        direction: str,
        duration: float = 0.5,
    ) -> str:
        """Move the head in a direction.

        Args:
            direction: One of 'left', 'right', 'up', 'down', 'center'
            duration: Movement duration in seconds

        Returns:
            Status message
        """
        # Define head poses for each direction (roll, pitch, yaw in degrees)
        poses = {
            "left": (0, 0, 25),      # yaw left
            "right": (0, 0, -25),    # yaw right
            "up": (0, -20, 0),       # pitch up
            "down": (0, 20, 0),      # pitch down
            "center": (0, 0, 0),     # neutral
        }

        if direction not in poses:
            logger.warning(f"Unknown direction: {direction}, using center")
            direction = "center"

        roll, pitch, yaw = poses[direction]

        # Create head pose matrix
        pose = create_head_pose(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            degrees=True,
        )

        logger.info(f"Moving head {direction}")
        await self._goto_target(head=pose, duration=duration)
        return f"Moved head {direction}"

    async def move_head_precise(
        self,
        roll: float = 0,
        pitch: float = 0,
        yaw: float = 0,
        duration: float = 0.5,
    ) -> str:
        """Move the head to a precise orientation.

        Args:
            roll: Roll angle in degrees (-30 to 30). Positive = tilt right
            pitch: Pitch angle in degrees (-30 to 30). Positive = look down
            yaw: Yaw angle in degrees (-45 to 45). Positive = look right
            duration: Movement duration in seconds

        Returns:
            Status message
        """
        # Clamp values to safe ranges
        roll = max(-30, min(30, roll))
        pitch = max(-30, min(30, pitch))
        yaw = max(-45, min(45, yaw))

        pose = create_head_pose(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            degrees=True,
        )

        logger.info(f"Moving head to roll={roll}, pitch={pitch}, yaw={yaw}")
        await self._goto_target(head=pose, duration=duration)
        return f"Moved head to roll={roll}, pitch={pitch}, yaw={yaw}"

    async def move_antennas(
        self,
        right_angle: float = 0,
        left_angle: float = 0,
        duration: float = 0.3,
    ) -> str:
        """Move the antennas to specific angles.

        Args:
            right_angle: Right antenna angle in degrees (-90 to 90)
            left_angle: Left antenna angle in degrees (-90 to 90)
            duration: Movement duration in seconds

        Returns:
            Status message
        """
        # Convert degrees to radians and clamp to safe range
        right_rad = max(-1.57, min(1.57, np.radians(right_angle)))
        left_rad = max(-1.57, min(1.57, np.radians(left_angle)))

        logger.info(f"Moving antennas: right={right_angle}, left={left_angle}")
        await self._goto_target(antennas=[right_rad, left_rad], duration=duration)
        return f"Moved antennas to right={right_angle}, left={left_angle} degrees"

    async def antenna_expression(self, expression: str) -> str:
        """Set antennas to a preset expression.

        Args:
            expression: One of 'neutral', 'alert', 'droopy', 'asymmetric', 'perky'

        Returns:
            Status message
        """
        expressions = {
            "neutral": [0, 0],
            "alert": [0.8, -0.8],      # Both up/forward
            "droopy": [-1.0, 1.0],     # Both down
            "asymmetric": [0.5, 0.2],  # One up, one down
            "perky": [1.0, -1.0],      # Both fully up
        }

        if expression not in expressions:
            expression = "neutral"

        antennas = expressions[expression]
        logger.info(f"Setting antenna expression: {expression}")
        await self._goto_target(antennas=antennas, duration=0.3)
        return f"Set antennas to {expression}"

    async def nod_yes(self, times: int = 2) -> str:
        """Nod head up and down (yes gesture).

        Args:
            times: Number of nods (1-5)

        Returns:
            Status message
        """
        times = max(1, min(5, times))
        logger.info(f"Nodding yes {times} times")

        for _ in range(times):
            # Nod down
            pose = create_head_pose(pitch=15, degrees=True)
            await self._goto_target(head=pose, duration=0.15)
            await asyncio.sleep(0.1)

            # Nod up
            pose = create_head_pose(pitch=-10, degrees=True)
            await self._goto_target(head=pose, duration=0.15)
            await asyncio.sleep(0.1)

        # Return to center
        pose = create_head_pose(degrees=True)
        await self._goto_target(head=pose, duration=0.2)
        return f"Nodded yes {times} times"

    async def shake_no(self, times: int = 2) -> str:
        """Shake head left and right (no gesture).

        Args:
            times: Number of shakes (1-5)

        Returns:
            Status message
        """
        times = max(1, min(5, times))
        logger.info(f"Shaking no {times} times")

        for _ in range(times):
            # Turn left
            pose = create_head_pose(yaw=20, degrees=True)
            await self._goto_target(head=pose, duration=0.15)
            await asyncio.sleep(0.05)

            # Turn right
            pose = create_head_pose(yaw=-20, degrees=True)
            await self._goto_target(head=pose, duration=0.15)
            await asyncio.sleep(0.05)

        # Return to center
        pose = create_head_pose(degrees=True)
        await self._goto_target(head=pose, duration=0.2)
        return f"Shook head no {times} times"

    async def tilt_head(self, direction: str, angle: float = 20) -> str:
        """Tilt head to one side (curious/quizzical gesture).

        Args:
            direction: 'left' or 'right'
            angle: Tilt angle in degrees (5-30)

        Returns:
            Status message
        """
        angle = max(5, min(30, angle))
        if direction == "left":
            roll = angle
        else:
            roll = -angle

        pose = create_head_pose(roll=roll, degrees=True)
        logger.info(f"Tilting head {direction} by {angle} degrees")
        await self._goto_target(head=pose, duration=0.4)
        return f"Tilted head {direction}"

    async def look_at_camera(self) -> str:
        """Look directly at the camera (center position)."""
        pose = create_head_pose(degrees=True)
        await self._goto_target(head=pose, antennas=[0, 0], duration=0.3)
        return "Looking at camera"

    async def wake_up(self) -> str:
        """Perform wake up animation."""
        logger.info("Performing wake up animation")
        try:
            await asyncio.to_thread(self.robot.wake_up)
            return "Woke up and ready!"
        except Exception as e:
            logger.error(f"Wake up error: {e}")
            return f"Wake up failed: {e}"

    async def go_to_sleep(self) -> str:
        """Perform sleep animation."""
        logger.info("Going to sleep")
        try:
            await asyncio.to_thread(self.robot.goto_sleep)
            return "Going to sleep..."
        except Exception as e:
            logger.error(f"Sleep error: {e}")
            return f"Sleep failed: {e}"

    async def express_emotion(self, emotion: str) -> str:
        """Express an emotion through movement.

        Args:
            emotion: One of 'happy', 'sad', 'surprised', 'curious',
                     'excited', 'sleepy', 'confused', 'angry', 'love'

        Returns:
            Status message
        """
        logger.info(f"Expressing emotion: {emotion}")

        emotion_methods = {
            "happy": self._happy_expression,
            "sad": self._sad_expression,
            "surprised": self._surprised_expression,
            "curious": self._curious_expression,
            "excited": self._excited_expression,
            "sleepy": self._sleepy_expression,
            "confused": self._confused_expression,
            "angry": self._angry_expression,
            "love": self._love_expression,
        }

        if emotion in emotion_methods:
            await emotion_methods[emotion]()
            return f"Expressed {emotion}"
        else:
            logger.warning(f"Unknown emotion: {emotion}")
            return f"Unknown emotion: {emotion}"

    async def _happy_expression(self) -> None:
        """Express happiness with a head wiggle and antenna bounce."""
        # Quick head tilt right
        pose = create_head_pose(roll=15, degrees=True)
        await self._goto_target(head=pose, antennas=[0.5, -0.5], duration=0.2)
        await asyncio.sleep(0.15)

        # Quick head tilt left
        pose = create_head_pose(roll=-15, degrees=True)
        await self._goto_target(head=pose, antennas=[-0.5, 0.5], duration=0.2)
        await asyncio.sleep(0.15)

        # Back to center with perky antennas
        pose = create_head_pose(degrees=True)
        await self._goto_target(head=pose, antennas=[0.3, -0.3], duration=0.2)

    async def _sad_expression(self) -> None:
        """Express sadness with droopy head and antennas."""
        # Look down with droopy antennas
        pose = create_head_pose(pitch=25, degrees=True)
        await self._goto_target(head=pose, antennas=[-1.2, 1.2], duration=0.8)
        await asyncio.sleep(0.8)

        # Slowly return to slightly droopy neutral
        pose = create_head_pose(pitch=5, degrees=True)
        await self._goto_target(head=pose, antennas=[-0.3, 0.3], duration=0.5)

    async def _surprised_expression(self) -> None:
        """Express surprise with quick look up and antenna pop."""
        # Quick look up with antennas up
        pose = create_head_pose(pitch=-20, degrees=True)
        await self._goto_target(head=pose, antennas=[1.0, -1.0], duration=0.12)
        await asyncio.sleep(0.4)

        # Return to neutral with slightly raised antennas
        pose = create_head_pose(degrees=True)
        await self._goto_target(head=pose, antennas=[0.3, -0.3], duration=0.3)

    async def _curious_expression(self) -> None:
        """Express curiosity with head tilt."""
        # Tilt head to side with one antenna up
        pose = create_head_pose(roll=20, pitch=-10, degrees=True)
        await self._goto_target(head=pose, antennas=[0.6, 0.1], duration=0.4)
        await asyncio.sleep(0.5)

        # Return to neutral
        pose = create_head_pose(degrees=True)
        await self._goto_target(head=pose, antennas=[0, 0], duration=0.3)

    async def _excited_expression(self) -> None:
        """Express excitement with bouncy movements."""
        for _ in range(3):
            # Quick up
            pose = create_head_pose(pitch=-10, degrees=True)
            await self._goto_target(head=pose, antennas=[0.8, -0.8], duration=0.1)
            await asyncio.sleep(0.08)

            # Quick down
            pose = create_head_pose(pitch=5, degrees=True)
            await self._goto_target(head=pose, antennas=[-0.2, 0.2], duration=0.1)
            await asyncio.sleep(0.08)

        # End with perky pose
        pose = create_head_pose(degrees=True)
        await self._goto_target(head=pose, antennas=[0.5, -0.5], duration=0.2)

    async def _sleepy_expression(self) -> None:
        """Express sleepiness with slow droopy movement."""
        # Slowly droop head and antennas
        pose = create_head_pose(pitch=25, roll=8, degrees=True)
        await self._goto_target(head=pose, antennas=[-1.3, 1.3], duration=1.2)
        await asyncio.sleep(0.5)

        # Small "nod off" movement
        pose = create_head_pose(pitch=30, roll=8, degrees=True)
        await self._goto_target(head=pose, duration=0.3)
        await asyncio.sleep(0.3)

        # Wake up slightly
        pose = create_head_pose(pitch=15, degrees=True)
        await self._goto_target(head=pose, antennas=[-0.8, 0.8], duration=0.4)

    async def _confused_expression(self) -> None:
        """Express confusion with head tilts and asymmetric antennas."""
        # Tilt one way
        pose = create_head_pose(roll=15, pitch=-5, degrees=True)
        await self._goto_target(head=pose, antennas=[0.4, 0.6], duration=0.3)
        await asyncio.sleep(0.3)

        # Tilt other way
        pose = create_head_pose(roll=-15, pitch=-5, degrees=True)
        await self._goto_target(head=pose, antennas=[0.6, 0.4], duration=0.3)
        await asyncio.sleep(0.3)

        # Return to slightly confused pose
        pose = create_head_pose(roll=8, degrees=True)
        await self._goto_target(head=pose, antennas=[0.2, 0.4], duration=0.25)

    async def _angry_expression(self) -> None:
        """Express anger with aggressive movements."""
        # Look down intensely
        pose = create_head_pose(pitch=10, degrees=True)
        await self._goto_target(head=pose, antennas=[0.8, -0.8], duration=0.2)
        await asyncio.sleep(0.2)

        # Quick shake
        pose = create_head_pose(yaw=10, pitch=10, degrees=True)
        await self._goto_target(head=pose, duration=0.1)
        await asyncio.sleep(0.1)

        pose = create_head_pose(yaw=-10, pitch=10, degrees=True)
        await self._goto_target(head=pose, duration=0.1)
        await asyncio.sleep(0.1)

        # Return to stern pose
        pose = create_head_pose(pitch=5, degrees=True)
        await self._goto_target(head=pose, antennas=[0.5, -0.5], duration=0.2)

    async def _love_expression(self) -> None:
        """Express love/affection with gentle movements."""
        # Gentle tilt with soft antenna pose
        pose = create_head_pose(roll=12, pitch=-8, degrees=True)
        await self._goto_target(head=pose, antennas=[0.4, -0.4], duration=0.5)
        await asyncio.sleep(0.4)

        # Other side
        pose = create_head_pose(roll=-12, pitch=-8, degrees=True)
        await self._goto_target(head=pose, antennas=[-0.4, 0.4], duration=0.5)
        await asyncio.sleep(0.4)

        # Return to happy neutral
        pose = create_head_pose(degrees=True)
        await self._goto_target(head=pose, antennas=[0.2, -0.2], duration=0.3)

    async def do_dance(self, style: str = "default") -> str:
        """Perform a short dance animation.

        Args:
            style: Dance style - 'default', 'happy', 'silly'

        Returns:
            Status message
        """
        logger.info(f"Dancing: {style}")

        if style == "silly":
            # Silly dance with exaggerated movements
            for _ in range(2):
                pose = create_head_pose(roll=25, yaw=15, degrees=True)
                await self._goto_target(head=pose, antennas=[1.0, 0], duration=0.2)
                await asyncio.sleep(0.15)

                pose = create_head_pose(roll=-25, yaw=-15, degrees=True)
                await self._goto_target(head=pose, antennas=[0, -1.0], duration=0.2)
                await asyncio.sleep(0.15)
        else:
            # Default/happy dance
            for _ in range(3):
                pose = create_head_pose(roll=15, pitch=-5, degrees=True)
                await self._goto_target(head=pose, antennas=[0.6, -0.6], duration=0.15)
                await asyncio.sleep(0.1)

                pose = create_head_pose(roll=-15, pitch=-5, degrees=True)
                await self._goto_target(head=pose, antennas=[-0.6, 0.6], duration=0.15)
                await asyncio.sleep(0.1)

        # Return to neutral
        pose = create_head_pose(degrees=True)
        await self._goto_target(head=pose, antennas=[0, 0], duration=0.25)
        return f"Finished {style} dance"

    async def reset_position(self) -> str:
        """Reset head and antennas to neutral position."""
        pose = create_head_pose(degrees=True)
        await self._goto_target(head=pose, antennas=[0, 0], duration=0.5)
        return "Reset to neutral position"
