from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the saved Random Forest model and label encoder
model = pickle.load(open('random_forest_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

app = Flask(__name__)

# Home route to render the home.html
@app.route('/')
def home():
    return render_template('home.html')

# Prediction route to handle user input and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form (only the 4 relevant features)
    data1 = float(request.form['clump-thickness'])  
    data2 = float(request.form['uniform-cell-size'])
    data3 = float(request.form['uniform-cell-shape'])
    data4 = float(request.form['marginal-adhesion'])
    
    # Create a numpy array from the user input (only 4 features)
    arr = np.array([[data1, data2, data3, data4]])
    
    # Predict using the trained Random Forest model
    pred = model.predict(arr)
    
    # Convert the prediction from binary to the original class using the label encoder
    decoded_prediction = label_encoder.inverse_transform(pred)

    # Render the result in the after.html template
    return render_template('after.html', data=decoded_prediction[0])  # Show the first (and only) prediction

if __name__ == "__main__":
    app.run(debug=True)

