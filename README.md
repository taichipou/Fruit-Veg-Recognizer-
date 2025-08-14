# Fruit-Veg-Recognizer
This project lets NVIDIA Jetson look at a photo and tell what fruit or vegetable is in it. The AI guesses the name and the confidence. It then writes that info right on the picture and saves a new JPG file for you. Every time you run it, you get a fresh, labeled image.


![](image-1.png)[](image.png](image-2.png))

## The Algorithm
After the AI learned the unique features of each fruit and vegetable by training on many fruit and vegetable images, the AI loads an image with jetson_utils.loadImage(), initializes jetson_inference.imageNet with ResNet-18 model and optional --model/--labels, classifies to get the top class and confidence, overlays filename | network | class (confidence) using cudaFont, then saves the labeled JPG and prints a short summary.

## Running this project

1. Login into your Nano and open up your python terminal.
2. Click on code and download as a .zip file, upload and unzip in nano under the jetson-inference/python/training/classification directory
3. change directory back to jetson-inference(command:cd jetson-inference)
4. run the docker(command:docker/run.sh)
5. cd into classification(command:cd python/training/classification)
6. 
7. Start training the network by passing the dataset(command: python3 train.py --model-dir=models/fruit_and_vegetable --epochs 80 --lr0.005 data/fruit_and_vegetable).The number of epochs and the learning rate are set to 80 and 0.005, respectively, to improve accuracy.
[View a video explanation here](video link)

```python
print(114514)
```
