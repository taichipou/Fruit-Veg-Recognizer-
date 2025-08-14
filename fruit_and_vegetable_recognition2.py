#!/usr/bin/env python3
import argparse, os, sys, datetime
import jetson_inference
import jetson_utils

parser = argparse.ArgumentParser(description="Fruit & Vegetable Recognition on Jetson")
parser.add_argument("filename", type=str, help="path to the image file to classify")
parser.add_argument("--out", type=str, default=None,
                    help="output jpg path (default: <input>_labeled.jpg)")
# Parse known so extra args can be passed through to imageNet (e.g., --model, --labels, --input_blob)
args, unknown = parser.parse_known_args()

# 1) load image
img = jetson_utils.loadImage(args.filename)
if img is None:
    print(f"[ERROR] failed to load image '{args.filename}'")
    sys.exit(1)

# 2) init network (pass any extra CLI args through)
net = jetson_inference.imageNet(argv=[sys.argv[0]] + unknown)
if not net:
    print(f"[ERROR] failed to initialize imageNet")
    sys.exit(1)

# 3) classify
class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)
class_network = net.GetNetworkName()

# 4) overlay text using CUDA font
#    (OverlayText(image, x, y, text, color=(r,g,b,a), background=(r,g,b,a)))
font = jetson_utils.cudaFont()

# Combine all information into a single, comprehensive line
combined_line = f"{os.path.basename(args.filename)} | {class_network} | {class_desc} ({confidence * 100:.2f}%)"

# Draw the combined line near the top-left
font.OverlayText(img, 10, 10, combined_line, color=(255,255,255,255), background=(0,0,0,160))

# 5) choose output path and save JPG
if args.out is None:
    root, _ = os.path.splitext(args.filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = f"{root}_labeled_{timestamp}.jpg"
else:
    out_root, out_ext = os.path.splitext(args.out)
    out_path = args.out if out_ext.lower() == ".jpg" else f"{out_root}.jpg"

jetson_utils.saveImage(out_path, img)

# 6) print a neat summary
print(f"Image:   {args.filename}")
print(f"Saved:   {out_path}")
print(f"Network: {class_network}")
print(f"Classified as '{class_desc}' (class #{class_idx}) with {confidence * 100:.2f}% confidence")