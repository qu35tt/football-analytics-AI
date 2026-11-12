from threading import Thread

import cv2

import main as main


WINDOW_NAME = "SoccerNet tracking"
FRAME_DELAY_MS = 30


def run_engine() -> None:
    """Run the unchanged tracking engine from main.py."""
    main.track_video()


def main_gui() -> None:
    output_dir = main.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    # Remove old engine output so the GUI shows this run only.
    for frame_path in output_dir.glob("frame_*.jpg"):
        frame_path.unlink()

    engine_thread = Thread(target=run_engine, daemon=True)
    engine_thread.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    frame_number = 0
    paused = False

    try:
        while engine_thread.is_alive() or (output_dir / f"frame_{frame_number:06d}.jpg").exists():
            frame_path = output_dir / f"frame_{frame_number:06d}.jpg"

            if not paused and frame_path.exists():
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    cv2.imshow(WINDOW_NAME, frame)
                    print(f"showing frame {frame_number}")
                    frame_number += 1

            key = cv2.waitKey(FRAME_DELAY_MS if not paused else 0) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main_gui()
