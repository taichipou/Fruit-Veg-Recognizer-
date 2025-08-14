Fruit-Veg-Recognizer
This project lets NVIDIA Jetson look at a photo and tell what fruit or vegetable is in it. The AI guesses the name and the confidence. It then writes that info right on the picture and saves a new JPG file for you. Every time you run it, you get a fresh, labeled image.



The Algorithm
After the AI learned the unique features of each fruit and vegetable by training on many fruit and vegetable images, the AI loads an image with jetson_utils.loadImage(), initializes jetson_inference.imageNet with ResNet-18 model and optional --model/--labels, classifies to get the top class and confidence, overlays filename | network | class (confidence) using cudaFont, then saves the labeled JPG and prints a short summary.

Running this project
Login into your Nano and open up your python terminal.
Click on code and download as a .zip file, upload and unzip in nano under the jetson-inference/python/training/classification directory
change directory back to jetson-inference(command:cd jetson-inference)
run the docker(command:docker/run.sh)
cd into classification(command:cd python/training/classification)
Start training the network by passing the dataset(command: python3 train.py --model-dir=models/fruit_and_vegetable --epochs 80 --lr0.005 data/fruit_and_vegetable).The number of epochs and the learning rate are set to 80 and 0.005, respectively, to improve accuracy.
Run fruit_and_vegetable_recognition2.py(command:python3 fruit_and_vegetable_recognition2.py Image.png
--model=./resnet18.onnx --labels=./labels.txt --input_blob=input_0 --output_blob=output_0)
The predicted label and its confidence will be shown on the image.
Required libraries to install jetson_inference jetson_utils argparse os sys datetime

[View a video explanation here](video lin
