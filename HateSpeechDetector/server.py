from flask import Flask, render_template, request
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K



app = Flask(__name__)

# Loading tokenizer
with open('tokenizer-2.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Define your custom metrics
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


def predict(text):
    # Pre-processing text (tokenizing)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=24)

    # predicting
    return model.predict(padded_sequence)

# Calculating the score (weighted average) of the prediction
def calculate_score(prediction):
    # 10 multiplier for hate speech, 5 multiplier for offensive speech, 0 multiplier for neither
    multipliers = [100, 50, 0]
    score = np.dot(prediction, multipliers)
    return round(score, 2)

# Classifying the text based on score 
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

@app.route('/', methods=['GET', 'POST'])
def home():
    classification = None
    text = ""
    score = None
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict(text)
        score = calculate_score(prediction[0])
        classification = classify_text(score)
    return render_template('index.html', prediction=classification, input_text=text, percent=score)

if __name__ == '__main__':
    app.run(debug=True)
