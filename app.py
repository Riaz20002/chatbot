from flask import Flask, render_template, request, jsonify
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained conversational model
chatbot = pipeline('text-generation', model='microsoft/DialoGPT-medium')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response_endpoint():
    user_message = request.json['message']
    if user_message:
        # Generate a response using the chatbot model
        response = chatbot(user_message, max_length=100, num_return_sequences=1)
        bot_reply = response[0]['generated_text']
        return jsonify({'response': bot_reply})
    else:
        return jsonify({'response': 'Please enter a valid message.'})

if __name__ == "__main__":
    app.run(debug=True)
