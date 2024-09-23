from flask import Flask, render_template, request, jsonify
import nltk
nltk.data.path.append('/opt/render/project/src/nltk_data')  # Add this line
from utils import predict_class, get_response  # Importing from utils.py

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_message', methods=['POST'])
def handle_message():
    data = request.get_json()
    message = data.get('message')
    if message:
        # Get the predicted intent(s) from the model
        intents = predict_class(message)
        if intents:  # Ensure there's a valid intent predicted
            # Get the response based on the predicted intent
            response = get_response(intents)
        else:
            response = "I didn't understand that. Can you please rephrase?"
        return jsonify({'response': response})
    return jsonify({'error': 'No message provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
