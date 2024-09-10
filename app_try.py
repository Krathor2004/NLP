from flask import Flask, logging, request, jsonify
import joblib
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
model_path = os.path.join(os.getcwd(), 'model', 'class.pkl')
classifier = joblib.load(model_path)

def predictfunc(reviews):
    predictions = classifier.predict(reviews)

    # Convert predictions to integers if they are returned as strings
    predict = int(predictions[0])

    # Debugging: print the raw predictions
    #logging.info("Raw predictions: %s", predictions)

    return predict  # Return the array of predictions

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    message = data.get('message', '')

    if message:
        review = pd.Series(message)
        prediction = predictfunc(review)
        # Prepare response
        if prediction == 0:
            is_bully = True
        else:
            is_bully = False
        print("bully",is_bully)        
        response = {
            'prediction': prediction,  # 0 or 1
            'is_bullying': is_bully  
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'No message provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Explicitly set port if needed
