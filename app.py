from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model (adjust the path as needed)

model = pickle.load(open('ML.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():

    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Render the prediction result
    return render_template('index.html', prediction_text=f'Diabetes Prediction: {"Positive" if prediction[0] == 1 else "Negative"}')

if __name__ == "__main__":
    app.run(debug=True)
