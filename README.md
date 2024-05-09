# hate-speech-detection
## Final project for CMSC421
## Group Members: Anirudh Poruri, Rahul Nair, Shreya Shete, Varun Chilukuri

### Overview
This repository contains all the code for our group's CMSC421 final project. For this project, we built a bi-directional long short-term memory RNN to detect offensive language and hate speech. There are three main components to this repository:
1. The data pre-processing, model definition, and model training notebook (**HateSpeechModel.ipynb**) - tweets.csv is the dataset used
2. The flask-server for the text classifier (**HateSpeechDetection** folder)
3. The flask-server to analyze YouTube Comment Sections (**Enter Directory** folder)

### Running text classifier flask app
1. `cd` into the `HateSpeechDetection` folder
2. `pip3 install -r requirements.txt` or `pip install -r requirements.txt`
3. `python3 server.py` or `python server.py`

### Running YouTube comment section analyzer flask app
1. `cd` into the `foldername` folder
2. `pip3 install -r requirements.txt` or `pip install -r requirements.txt`
3. `python3 server.py` or `python server.py`
