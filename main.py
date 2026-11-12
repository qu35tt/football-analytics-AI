from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from engine.tracker import PlayerGrouper, upper_torso_crop
from ultralytics import YOLO


VIDEO_PATH = Path("video.mp4")
OUTPUT_DIR = Path("output")
MODEL_PATH = "runs/train/football-yolo11s-9/weights/best.pt"

# Custom dataset class IDs used for training:
# 0 = player, 1 = ball, 2 = referee
INTERESTING_CLASSES = {0, 1, 2}
PLAYER_CONFIDENCE = 0.15
BALL_CONFIDENCE = 0.05
TRAJECTORY_LENGTH = 30


def track_video() -> None:
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)
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

            # YOLO detects and tracks with native ByteTrack.
            results = model.track(
                frame,
                classes=list(INTERESTING_CLASSES),
                conf=BALL_CONFIDENCE,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )
            detections = sv.Detections.from_ultralytics(results[0])
            tracked_players = detections[
                (detections.class_id == 0)
                & (detections.confidence >= PLAYER_CONFIDENCE)
            ]
            tracked_balls = detections[
                (detections.class_id == 1)
                & (detections.confidence >= BALL_CONFIDENCE)
            ]

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
