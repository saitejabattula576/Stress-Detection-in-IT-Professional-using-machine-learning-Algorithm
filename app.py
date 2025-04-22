import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

# Function to store user credentials
def store_credentials(username, password):
    with open("credentials.txt", "a") as file:
        file.write(f"{username}:{password}\n")

# Function to authenticate user
def authenticate(username, password):
    with open("credentials.txt", "r") as file:
        for line in file:
            stored_username, stored_password = line.strip().split(":")
            if stored_username == username and stored_password == password:
                return True
    return False

# Function to check if a username exists
def check_username(username):
    with open("credentials.txt", "r") as file:
        for line in file:
            stored_username, _ = line.strip().split(":")
            if stored_username == username:
                return True
    return False

# Function for login page
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.success("You are now logged in as {}".format(username))
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password.")

# Function for signup page
def signup():
    st.title("Sign Up")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Sign Up"):
        if check_username(new_username):
            st.error("Username already exists.")
        else:
            store_credentials(new_username, new_password)
            st.success("You have successfully signed up as {}".format(new_username))

# Function for logout
def logout():
    st.session_state.logged_in = False

# Load the existing image-based stress detection model
def load_existing_model():
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    existing_model = model_from_json(loaded_model_json)
    existing_model.load_weights("model.h5")
    return existing_model

# Load the trained model for stress level prediction
def load_trained_model():
    model = joblib.load('stress_model.pkl')
    return model

# Preprocess image for prediction
def preprocess_image(img):
    img = img.resize((48, 48))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to detect stress from image
def detect_stress(img, model):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    threshold = 0.16
    predicted_class = "Not Stressed" if prediction[0][0] > threshold else "Stressed"
    recommendations = []
    if predicted_class == "Stressed":
        recommendations.append("Take a deep breath and try some relaxation techniques.")
        recommendations.append("Practice mindfulness or meditation regularly.")
        recommendations.append("Engage in physical activities like yoga or exercise.")
        recommendations.append("Seek support from friends, family, or a mental health professional.")
        recommendations.append("Limit exposure to stressful triggers and prioritize self-care.")
    return predicted_class, recommendations

# Function to predict stress level
def predict_stress_level(features, model):
    data = pd.DataFrame(features, index=[0])
    stress_level = model.predict(data)
    return stress_level[0]

# Main function for Streamlit app  
def main():
    st.title("Stress Detection for IT professionals")
    
    # Check if user is logged in
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        login()
        signup()
    else:
        st.sidebar.write("Welcome!")
        if st.sidebar.button("Logout"):
            logout()
    
        # Home page with options to select existing or new model
        st.subheader("Home Page")
        model_choice = st.radio("Select Model", ("Image-Based Stress Detection", "Stress Level Prediction"))

        if model_choice == "Image-Based Stress Detection":
            existing_model = load_existing_model()
            existing_model_page(existing_model)
        elif model_choice == "Stress Level Prediction":
            model = load_trained_model()
            new_model_page(model)

# Page for existing image-based stress detection model
def existing_model_page(model):
    st.write("You selected the Image-Based Stress Detection Model.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        prediction, recommendations = detect_stress(img, model)
        st.success(f"Prediction: {prediction}")
        if prediction == "Stressed":
            st.write("Recommendations:")
            for recommendation in recommendations:
                st.write("- " + recommendation)
                if "physical activities" in recommendation:
                    st.write("Consider going for a walk or jog to reduce stress.")
                elif "meditation" in recommendation:
                    st.write("Try a guided meditation session to calm your mind.")
                # Add more specific recommendations and corresponding actions as needed
        else:
            st.write("Stay stress-free!")

# Page for stress level prediction based on input features
def new_model_page(model):
    st.write("You selected the Stress Level Prediction Model.")

    # Set default minimum and maximum values for sliders
    min_sleep_quality = 1
    max_sleep_quality = 10
    min_headache_frequency = 0
    max_headache_frequency = 7
    min_work_load = 1
    max_work_load = 10
    min_extracurricular_activities = 0
    max_extracurricular_activities = 7

    sleep_quality = st.slider('Rate your sleep quality', min_value=min_sleep_quality, max_value=max_sleep_quality, step=1)
    headache_frequency = st.slider('How many times a week do you suffer headaches?', min_value=min_headache_frequency, max_value=max_headache_frequency, step=1)
    work_load = st.slider('Rate your work load', min_value=min_work_load, max_value=max_work_load, step=1)
    extracurricular_activities = st.slider('How many times a week do you practice extracurricular activities?', min_value=min_extracurricular_activities, max_value=max_extracurricular_activities, step=1)

    if st.button('Predict Stress Level'):
        input_features = {
            'kindly_rate_your_sleep_quality_😴': sleep_quality,
            'how_many_times_a_week_do_you_suffer_headaches_🤕?': headache_frequency,
            'how_would_you_rate_your_work_load?': work_load,
            'how_many_times_a_week_you_practice_extracurricular_activities_🎾?': extracurricular_activities
        }
        
        stress_level = predict_stress_level(input_features, model)
        
        # Function to interpret stress level
        def interpret_stress_level(stress_level):
            if stress_level < 3:
                return "Low Stress"
            elif stress_level < 5:
                return "Moderate Stress"
            else:
                return "High Stress"

        # Interpret stress level
        interpreted_stress_level = interpret_stress_level(stress_level)
        st.write(f'Predicted Stress Level: {interpreted_stress_level} ({stress_level:.2f})')
        
        # Provide mitigation strategies based on stress level
        if interpreted_stress_level == "High Stress":
            st.write("It seems like you're experiencing high stress. Here are some strategies to help you manage it:")
            st.write("- Consider seeking support from a mental health professional.")
            st.write("- Practice deep breathing exercises or meditation regularly.")
            st.write("- Make time for activities you enjoy and prioritize self-care.")
            # Add more specific recommendations for high stress level
        elif interpreted_stress_level == "Moderate Stress":
            st.write("You're experiencing moderate stress. Here are some strategies to help you cope with it:")
            st.write("- Engage in regular physical activities to reduce stress.")
            st.write("- Try relaxation techniques like yoga or mindfulness meditation.")
            st.write("- Set realistic goals and prioritize tasks to manage your workload.")
            # Add more specific recommendations for moderate stress level
        else:
            st.write("Your stress level seems low. Keep up with healthy habits and self-care practices to maintain it.")
            # Add more specific recommendations for low stress level

if __name__ == "__main__":
    main()
