# Introduction

This Project is based on openCV and deep learning which recognize a person's emotion standing in front of camera. There are total seven emotions included in this project with their respective labels: 0 : 'Angry' , 1 : 'Disgust' , 2 : 'Fear' , 3 : 'Happy' , 4 : 'Sad' , 5 : 'Surprise' , 6 : 'Neutral'.

## Requirements
1)OpenCV

2)Tensorflow/keras for training CNN model.

## Installation

Use the package manager pip to install opencv.

```bash
pip install opencv-python
```

## Steps for running this project

1) For generating the dataset in cleaned format run generate_data.py

2) For training our CNN model, run trainer.py

3) Run the emotion_predictor.py file in command line.

## More about files

The dataset of this project is available on [kaggle fe2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). This is untidy dataset . For cleaning and generating a new dataframe , run generate_data.py.

## Contribution

Thanks to Nikita Pande and Zainab Nomi : https://github.com/zainabnomi , for valuable contribution.

## Sample image of final result

![sample_image](project.jpeg)
