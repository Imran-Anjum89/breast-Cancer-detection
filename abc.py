import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from datetime import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Target variable (0 = Malignant, 1 = Benign)

# Select the 10 most important features using Recursive Feature Elimination (RFE)
model_for_rfe = LogisticRegression(random_state=42, max_iter=1000)
rfe = RFE(estimator=model_for_rfe, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
selected_features = data.feature_names[rfe.support_]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model using the selected features
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Function to handle predictions
def make_prediction():
    try:
        # Collect user input for selected features
        user_input = []
        for feature in selected_features:
            value = float(entry_dict[feature].get())  # Get value from input fields
            user_input.append(value)

        # Convert user input to NumPy array and scale it
        user_input = np.array(user_input).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)

        # Make a prediction
        prediction = model.predict(user_input_scaled)
        prediction_prob = model.predict_proba(user_input_scaled)

        # Show result in Tkinter window
        if prediction[0] == 0:
            result = "Malignant (cancerous)"
        else:
            result = "Benign (non-cancerous)"

        confidence = f"Confidence: {prediction_prob[0][prediction[0]] * 100:.2f}%"

        # Display result in Tkinter window
        global prediction_result
        prediction_result = f"Prediction: {result}\n{confidence}"

        # Show the third page
        show_third_page()

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for all features.")

# Create the main Tkinter window
root = tk.Tk()
root.title("Breast Cancer Prediction")

# Allow the window to be resized
root.resizable(True, True)

# Define global variables for storing personal details
personal_details = {}

# --- Front Page ---
def show_front_page():
    clear_window()
    
    # Title and Instructions
    title_label = tk.Label(root, text="Breast Cancer Prediction", font=("Arial", 24), bg="#5b9bd5", fg="white")
    title_label.grid(row=0, column=0, columnspan=2, pady=20)
    
    # Personal Details Form
    tk.Label(root, text="Name", font=("Arial", 14)).grid(row=1, column=0, padx=10, pady=5)
    name_entry = tk.Entry(root, font=("Arial", 14))
    name_entry.grid(row=1, column=1, padx=10, pady=5)
    
    tk.Label(root, text="Age", font=("Arial", 14)).grid(row=2, column=0, padx=10, pady=5)
    age_entry = tk.Entry(root, font=("Arial", 14))
    age_entry.grid(row=2, column=1, padx=10, pady=5)
    
    tk.Label(root, text="Sex (M/F)", font=("Arial", 14)).grid(row=3, column=0, padx=10, pady=5)
    sex_entry = tk.Entry(root, font=("Arial", 14))
    sex_entry.grid(row=3, column=1, padx=10, pady=5)

    # Current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tk.Label(root, text=f"Date and Time: {current_time}", font=("Arial", 12)).grid(row=4, column=0, columnspan=2, pady=5)
    
    def validate_inputs():
        name = name_entry.get()
        age = age_entry.get()
        sex = sex_entry.get()
        
        if not name or not age or not sex:
            messagebox.showerror("Input Error", "Please fill in all fields correctly.")
        else:
            personal_details["name"] = name
            personal_details["age"] = age
            personal_details["sex"] = sex
            personal_details["datetime"] = current_time
            show_second_page()

    # Button to proceed to next page
    next_button = tk.Button(root, text="Next", font=("Arial", 14), bg="#4CAF50", fg="white", command=validate_inputs)
    next_button.grid(row=5, column=0, columnspan=2, pady=20)

    # Set the 'Enter' key to click the "Next" button
    root.bind("<Return>", lambda event: next_button.invoke())

# --- Second Page (Breast Cancer Test) ---
def show_second_page():
    clear_window()

    # Title
    title_label = tk.Label(root, text="Breast Cancer Test", font=("Arial", 24), bg="#5b9bd5", fg="white")
    title_label.grid(row=0, column=0, columnspan=2, pady=20)
    
    global entry_dict
    entry_dict = {}

    # Input fields for breast cancer test features
    for idx, feature in enumerate(selected_features):
        label = tk.Label(root, text=feature, font=("Arial", 14))
        label.grid(row=idx + 1, column=0, padx=10, pady=5)

        entry = tk.Entry(root, font=("Arial", 14))
        entry.grid(row=idx + 1, column=1, padx=10, pady=5)
        entry_dict[feature] = entry

    # Button to perform test
    test_button = tk.Button(root, text="Test", font=("Arial", 14), bg="#4CAF50", fg="white", command=make_prediction)
    test_button.grid(row=len(selected_features) + 1, column=0, columnspan=2, pady=20)

    # Set the 'Enter' key to click the "Test" button
    root.bind("<Return>", lambda event: test_button.invoke())

# --- Third Page (Medical Report) ---
def show_third_page():
    clear_window()
    
    # Title
    title_label = tk.Label(root, text="Breast Cancer Test Report", font=("Arial", 24), bg="#5b9bd5", fg="white")
    title_label.grid(row=0, column=0, columnspan=2, pady=20)
    
    # Display personal details
    personal_info = f"Name: {personal_details['name']}\nAge: {personal_details['age']}\nSex: {personal_details['sex']}\nDate & Time: {personal_details['datetime']}"
    personal_label = tk.Label(root, text=personal_info, font=("Arial", 14), anchor="w", justify=tk.LEFT)
    personal_label.grid(row=1, column=0, columnspan=2, pady=20)

    # Display the result of the test (prediction)
    result_label = tk.Label(root, text=prediction_result, font=("Arial", 16), bg="#fffff0")
    result_label.grid(row=2, column=0, columnspan=2, pady=10)

    # Button to exit the application
    exit_button = tk.Button(root, text="Exit", font=("Arial", 14), bg="#f44336", fg="white", command=root.quit)
    exit_button.grid(row=3, column=0, columnspan=2, pady=20)

    # Set the 'Enter' key to click the "Exit" button
    root.bind("<Return>", lambda event: root.quit())

# --- Clear the current window ---
def clear_window():
    for widget in root.winfo_children():
        widget.destroy()

# Start with the front page
show_front_page()

root.mainloop()