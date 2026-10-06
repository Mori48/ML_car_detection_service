"""Inference throughput benchmark: detector + tracker only.

Excludes video decoding, JPEG encoding, network and UI, so the result
reflects the model pipeline and nothing else.

Usage (inside the backend container):
    python -m backend.benchmark --video /app/data/videos/test.mp4 \
        --weights /app/data/models/best.pt --device cpu
"""
import argparse
import statistics
import time

import cv2
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True, help=".pt or .engine file")
    p.add_argument("--device", default="0", help="'cpu' or GPU index, e.g. 0")
    p.add_argument("--imgsz", type=int, default=480)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    device = args.device if args.device == "cpu" else int(args.device)
    model = YOLO(args.weights, task="detect")

    # Read and resize frames up front so decoding is not part of the timing.
    cap = cv2.VideoCapture(args.video)
    needed = args.frames + args.warmup
    frames = []
    while len(frames) < needed:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (854, 480)))
    cap.release()

    if len(frames) <= args.warmup:
        raise SystemExit(f"Video too short: {len(frames)} frames read")

    times = []
    for i, frame in enumerate(frames):
        start = time.perf_counter()
        model.track(
            source=frame,
            persist=True,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            max_det=35,
            verbose=False,
            tracker="botsort.yaml",
            agnostic_nms=True,
            classes=[0, 1, 3],
        )
        elapsed = time.perf_counter() - start
        if i >= args.warmup:
            times.append(elapsed * 1000)

    times.sort()
    mean = statistics.mean(times)
    p95 = times[int(0.95 * (len(times) - 1))]
    print(f"weights:     {args.weights}")
    print(f"device:      {args.device}")
    print(f"frames:      {len(times)} (after {args.warmup} warm-up)")
    print(f"mean:        {mean:.1f} ms/frame")
    print(f"p95:         {p95:.1f} ms/frame")
    print(f"throughput:  {1000 / mean:.1f} FPS")


if __name__ == "__main__":
    main()