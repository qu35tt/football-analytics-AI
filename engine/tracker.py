import cv2
import numpy as np


TORSO_TOP = 0.15
TORSO_BOTTOM = 0.60
REFEREE_COLOR_RATIO = 0.15


def has_referee_colors(torso: np.ndarray) -> bool:
    """Check for black, yellow, lime, or dark sky-blue jersey pixels."""
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    black = (value <= 70) & (saturation <= 140)
    yellow = (
        (hue >= 18)
        & (hue <= 40)
        & (saturation >= 80)
        & (value >= 70)
    )
    lime = (
        (hue >= 35)
        & (hue <= 80)
        & (saturation >= 70)
        & (value >= 50)
    )
    dark_sky_blue = (
        (hue >= 85)
        & (hue <= 115)
        & (saturation >= 45)
        & (value >= 35)
        & (value <= 190)
    )
    referee_pixels = black | yellow | lime | dark_sky_blue
    return float(np.mean(referee_pixels)) >= REFEREE_COLOR_RATIO


def upper_torso_crop(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """Return the upper body area, excluding the head, shorts, and grass."""
    x1, y1, x2, y2 = box.astype(int)
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    torso_y1 = y1 + int((y2 - y1) * TORSO_TOP)
    torso_y2 = y1 + int((y2 - y1) * TORSO_BOTTOM)
    crop = frame[torso_y1:torso_y2, x1:x2]
    return crop if crop.size else None


class PlayerGrouper:
    """Groups players by torso appearance and keeps groups stable over time."""

    def __init__(self) -> None:
        self.team_colors: list[np.ndarray | None] = [None, None]
        self.player_groups: dict[int, str] = {}

    def group(self, player_id: int, torso: np.ndarray) -> str:
        if player_id in self.player_groups:
            return self.player_groups[player_id]

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        color = np.mean(hsv.reshape(-1, 3), axis=0)

        if has_referee_colors(torso):
            self.player_groups[player_id] = "referee"
            return "referee"

        available = [i for i, team_color in enumerate(self.team_colors) if team_color is None]
        if available:
            team_index = available[0]
        else:
            distances = [
                float(np.linalg.norm(color - team_color))
                for team_color in self.team_colors
                if team_color is not None
            ]
            team_index = int(np.argmin(distances))

        if self.team_colors[team_index] is None:
            self.team_colors[team_index] = color
        else:
            self.team_colors[team_index] = (
                self.team_colors[team_index] * 0.9 + color * 0.1
            )

        group_name = f"team_{'a' if team_index == 0 else 'b'}"
        self.player_groups[player_id] = group_name
        return group_name
