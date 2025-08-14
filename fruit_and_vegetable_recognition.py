#!/usr/bin/env python3

import sys
import argparse

import jetson_inference
import jetson_utils

def main():
    # 1) parse the one required positional argument (image filename)
    parser = argparse.ArgumentParser(
        description="Fruit & Vegetable Recognition on Jetson"
    )
    parser.add_argument(
        "filename",
        type=str,
        help="path to the image file to classify"
    )
    # collect any other flags (–model, –labels, –input_blob, –output_blob, –network, etc.)
    args, unknown = parser.parse_known_args()

    # 2) load the image from disk into GPU memory
    img = jetson_utils.loadImage(args.filename)
    if img is None:
        print(f"[ERROR] failed to load image '{args.filename}'")
        sys.exit(1)

    # 3) initialize your custom network:
    #    pass the full argv list so imageNet picks up –model, –labels, –input_blob, –output_blob, –threshold, –network, etc.
    net = jetson_inference.imageNet(argv=sys.argv)
    if not net:
        print(f"[ERROR] failed to initialize imageNet")
        sys.exit(1)

    # 4) run inference
    class_idx, confidence = net.Classify(img)
    class_desc       = net.GetClassDesc(class_idx)
    class_network    = net.GetNetworkName()

    # 5) print a neat summary
    print(f"Image:   {args.filename}")
    print(f"Network: {class_network}")
    print(f"Classified as '{class_desc}' (class #{class_idx}) "
          f"with {confidence * 100:.2f}% confidence")

if __name__ == "__main__":
    main()