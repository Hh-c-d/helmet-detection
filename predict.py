"""
Run the trained model on an image, a directory, or a video.
Annotated results are saved to the output directory, plus a detection
count summary (useful for customer handover).

Usage:
    python predict.py --weights best.pt --source site.jpg
    python predict.py --weights best.pt --source video.mp4 --out result.mp4
"""
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Custom detection inference")
    p.add_argument("--weights", required=True, help="path to best.pt")
    p.add_argument("--source", required=True, help="image, image dir, or video file")
    p.add_argument("--out", default="runs/helmet/predict", help="output dir (images) or file (video)")
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, save=True,
                            project=str(Path(args.out).parent), name=Path(args.out).name)
    total = Counter()
    for r in results:
        for cls in r.boxes.cls.cpu().numpy():
            total[model.names[int(cls)]] += 1
    print("Detection counts:")
    for name, n in total.most_common():
        print(f"  {name}: {n}")
    print(f"Saved results under {args.out}")


if __name__ == "__main__":
    main()
