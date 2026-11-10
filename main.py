from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


VIDEO_PATH = Path("video.mp4")
OUTPUT_DIR = Path("ouput")
MODEL_PATH = "yolo11s.pt"

# COCO class IDs used by the model:
# 0 = person, 32 = sports ball.
INTERESTING_CLASSES = {0, 32}
PLAYER_CONFIDENCE = 0.15
BALL_CONFIDENCE = 0.05
TRAJECTORY_LENGTH = 30
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


def track_video() -> None:
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack(
        track_activation_threshold=PLAYER_CONFIDENCE,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )
    ball_tracker = sv.ByteTrack(
        track_activation_threshold=BALL_CONFIDENCE,
        lost_track_buffer=60,
        minimum_matching_threshold=0.3,
        frame_rate=30,
    )
    trajectories: dict[tuple[str, int], list[tuple[int, int]]] = {}
    player_grouper = PlayerGrouper()
    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(f"OpenCV could not open {VIDEO_PATH}")

    frame_number = 0
    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            # YOLO detects players and the ball. Supervision assigns tracking IDs.
            results = model.predict(
                frame,
                classes=list(INTERESTING_CLASSES),
                conf=BALL_CONFIDENCE,
                verbose=False,
            )
            detections = sv.Detections.from_ultralytics(results[0])
            player_detections = detections[
                (detections.class_id == 0)
                & (detections.confidence >= PLAYER_CONFIDENCE)
            ]
            ball_detections = detections[
                (detections.class_id == 32)
                & (detections.confidence >= BALL_CONFIDENCE)
            ]
            tracked_players = tracker.update_with_detections(player_detections)
            tracked_balls = ball_tracker.update_with_detections(ball_detections)

            annotated_frame = frame.copy()
            tracked_ids = []
            for name, tracked, color in (
                ("player", tracked_players, (0, 255, 0)),
                ("ball", tracked_balls, (0, 165, 255)),
            ):
                if tracked.tracker_id is None:
                    continue
                for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
                    x1, y1, x2, y2 = box.astype(int)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    trajectory_key = (name, int(track_id))
                    trajectories.setdefault(trajectory_key, []).append(center)
                    trajectories[trajectory_key] = trajectories[trajectory_key][
                        -TRAJECTORY_LENGTH:
                    ]

                    if name == "player":
                        torso = upper_torso_crop(frame, box)
                        group = (
                            player_grouper.group(int(track_id), torso)
                            if torso is not None
                            else "unknown"
                        )
                        crop_dir = OUTPUT_DIR / "crops" / group
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        if torso is not None:
                            crop_path = crop_dir / f"player_{int(track_id)}_frame_{frame_number:06d}.jpg"
                            cv2.imwrite(str(crop_path), torso)
                    else:
                        group = "ball"
                    tracked_ids.append(f"{group}:{int(track_id)}")

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated_frame,
                        f"{group} ID:{int(track_id)}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            for points in trajectories.values():
                if len(points) > 1:
                    cv2.polylines(
                        annotated_frame,
                        [np.array(points, dtype=np.int32)],
                        False,
                        (255, 0, 255),
                        2,
                    )
            output_path = OUTPUT_DIR / f"frame_{frame_number:06d}.jpg"
            if not cv2.imwrite(str(output_path), annotated_frame):
                raise OSError(f"Could not write {output_path}")

            print(f"frame {frame_number}: tracking IDs {tracked_ids}")
            frame_number += 1
    finally:
        video.release()

    print(f"Saved {frame_number} tracked frames to {OUTPUT_DIR}/")


if __name__ == "__main__":
    track_video()
