import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load a sample dataset (replace with your own data if needed)
data = load_iris()
X = data.data
y = data.target

# Transform the labels
le = LabelBinarizer()
y = le.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# Save the trained model to a .pkl file
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Optionally, save the label encoder for later use
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

# Example of loading the model back and using it for predictions
# Load the model back
with open('random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as le_file:
    loaded_le = pickle.load(le_file)

# Make predictions with the loaded model
predictions = loaded_model.predict(X_test)

# If needed, you can convert the predicted labels back to original labels
decoded_predictions = loaded_le.inverse_transform(predictions)

print(decoded_predictions)
