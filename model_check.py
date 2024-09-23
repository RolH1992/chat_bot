from tensorflow.keras.models import load_model

try:
    model = load_model('model/chatbot_model.keras')
    print("Model loaded successfully.")
    print(model.summary())  # Display the model structure
except Exception as e:
    print(f"Error loading model: {e}")

    