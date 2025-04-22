import streamlit as st
import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model architecture from JSON file
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Load the model weights
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

def preprocess_image(img):
    img = img.resize((48, 48))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_stress(img):
    img_array = preprocess_image(img)

    # Make prediction
    prediction = model.predict(img_array)

    # Define the threshold
    threshold = 0.16
    predicted_class = "Not Stressed" if prediction[0][0] > threshold else "Stressed"

    # Recommendations based on prediction
    recommendations = []
    if predicted_class == "Stressed":
        recommendations.append("Take a deep breath and try some relaxation techniques.")
        recommendations.append("Practice mindfulness or meditation regularly.")
        recommendations.append("Engage in physical activities like yoga or exercise.")
        recommendations.append("Seek support from friends, family, or a mental health professional.")
        recommendations.append("Limit exposure to stressful triggers and prioritize self-care.")

    return predicted_class, recommendations

def main():
    st.title("Stress Detection and Management App")
    st.sidebar.image('giphy.gif')
    st.write("Upload an image to detect if it's stressed or not stressed.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Detect stress
        prediction, recommendations = detect_stress(img)
        st.success(f"Prediction: {prediction}")

        if prediction == "Stressed":
            st.write("Recommendations:")
            for recommendation in recommendations:
                st.write("- " + recommendation)
        else:
            st.write("Stay stress-free!")
if __name__ == "__main__":
    main()
