from flask import Flask, render_template, request, jsonify
import os
import nltk

nltk.data.path.append('/opt/render/project/src/nltk_data')
from utils import predict_class, get_response

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_message', methods=['POST'])
def handle_message():
    data = request.get_json()
    message = data.get('message')
    if message:
        intents = predict_class(message)
        if intents:
            response = get_response(intents)
        else:
            response = "I didn't understand that. Can you please rephrase?"
        return jsonify({'response': response})
    return jsonify({'error': 'No message provided'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
