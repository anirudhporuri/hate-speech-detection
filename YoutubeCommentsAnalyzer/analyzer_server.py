from flask import Flask, render_template, request
import os
import googleapiclient.discovery
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('tokenizer-2.json', 'r', encoding='utf-8') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Load the model and specify the custom objects
model = load_model('model-2.h5', custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})


# model = tf.keras.models.load_model('model-1.h5')

def calculate_score(predictions):
    multipliers = np.array([100, 50, 0])
    scores = np.dot(predictions, multipliers)
    return np.sum(scores)

def classify_text(score):
    if 80 <= score <= 100:
        return 'Hate Speech'
    elif 60 <= score < 80:
        return 'Very Offensive Language/Leaning Towards Hate Speech'
    elif 40 <= score < 60:
        return 'Offensive Language'
    elif 20 <= score < 40:
        return 'Leaning Towards Offensive Language'
    elif 0 <= score < 20:
        return 'Not Offensive or Hate Speech'
    else:
        return 'Unknown Category' 

def predict_hate_speech(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=24)
    return model.predict(padded_sequence)

def extract_video_id(youtube_link):
    parts = youtube_link.split('/')
    video_id_part = parts[-1]
    video_id = video_id_part.split('?')[0]
    return video_id

def get_video_name(video_id):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey="AIzaSyCExw8b06tMONT7x3IcUzA9iUBdJ1uDCMQ")

    response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    if 'items' in response and len(response['items']) > 0:
        return response['items'][0]['snippet']['title']
    else:
        return None

def get_youtube_comments(video_id, max_comments): 
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey="AIzaSyCExw8b06tMONT7x3IcUzA9iUBdJ1uDCMQ")

    comments = []
    scores_count = {'Hate Speech': 0,
                    'Very Offensive Language/Leaning Towards Hate Speech': 0,
                    'Offensive Language': 0,
                    'Leaning Towards Offensive Language': 0,
                    'Not Offensive or Hate Speech': 0,
                    'Unknown Category': 0}

    next_page_token = None
    total_comments = 0

    while total_comments < max_comments:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - total_comments), 
            pageToken=next_page_token
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            prediction = predict_hate_speech(comment)
            score = calculate_score(prediction)
            category = classify_text(score)
            scores_count[category] += 1
            total_comments += 1

            if total_comments >= max_comments:
                break

        if total_comments >= max_comments:
            break

        if 'nextPageToken' in response:
            next_page_token = response['nextPageToken']
        else:
            break

    return comments, scores_count

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_link = request.form['youtube_link']
        video_id = extract_video_id(youtube_link)
        print("Extracted Video ID:", video_id)
        return render_template('index.html', video_id=video_id)
    else:
        return render_template('index.html')

@app.route('/hate_scores', methods=['POST'])
def hate_scores():
    maxcomments = 100
    youtube_link = request.form['youtube_link']
    video_id = extract_video_id(youtube_link)
    print("Extracted Video ID:", video_id)
    video_name = get_video_name(video_id)
    comments, scores_count = get_youtube_comments(video_id, maxcomments) 
    num_comments = len(comments)
    percentage_scores = {category: (count / num_comments) * 100 for category, count in scores_count.items()}
    scores_count = percentage_scores
    total_score = 0
    for comment in comments:
        prediction = predict_hate_speech(comment)
        total_score += calculate_score(prediction)
    weighted_average_score = total_score / num_comments
    print("Num", num_comments)
    print("Max", maxcomments)

    return render_template('hate_scores.html', video_id=video_id, video_name=video_name,
                           num_comments=num_comments, offensive_comments_count=scores_count['Hate Speech'],
                           scores_count=scores_count, weighted_average=weighted_average_score, maxcomments=min(num_comments,maxcomments))

if __name__ == '__main__':
    app.run(debug=True)
