import os
import random
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, render_template, request, jsonify

# Ensure nltk resources are downloaded
nltk.download('punkt')

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')

# Function to clean up and tokenize sentences
def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    ignore_symbols = ['?', '!', '.', ',']
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word not in ignore_symbols]
    return sentence_words

# Function to create a bag of words
def bag_of_words(sentence):
    words = pickle.load(open('model/words.pkl', 'rb'))
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class of the sentence
def predict_class(sentence):
    classes = pickle.load(open('model/classes.pkl', 'rb'))
    model = load_model('model/chatbot_model.keras')
    
    # Print the model summary to verify if it's loaded correctly
    print(model.summary())  # This will show if the model structure is intact

    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    if return_list:
        print(f"Predicted intent: {return_list[0]['intent']} with probability {return_list[0]['probability']}")
    else:
        print("No intent could be predicted with sufficient confidence.")
    return return_list 

# Function to get the response from the intents JSON file
def get_response(intents_list):
    if not intents_list:
        return "I'm not sure how to respond to that. Can you ask me something else?"

    intents_json = json.load(open('model/intents.json'))
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            print(f"Selected response: {result}")  # Debugging print
            return result
    return "I'm sorry, I don't understand that."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_message', methods=['POST'])
def handle_message():
    message = request.json.get('message')
    if message:
        # Get the predicted intent(s) from the model
        intents = predict_class(message)
        if intents:
            # Get the response based on the predicted intent
            response = get_response(intents)
        else:
            response = "I didn't understand that. Can you please rephrase?"
        return jsonify({'response': response})
    return jsonify({'error': 'No message provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
