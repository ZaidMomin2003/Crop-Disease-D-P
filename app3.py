import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import google.generativeai as genai

# Configure Gemini AI
genai.configure(api_key="AIzaSyBoFhFSFz7VfEo5QdgGpyhixL-vRGq-3Qc")
model = genai.GenerativeModel("gemini-pro")

# Load the TensorFlow Lite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path='model_unquant.tflite')
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  
    image = np.array(image, dtype=np.float32) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# Decode predictions based on class labels
class_labels = [
    "Tomato Yellow Leaf Curl Virus", "Tomato Septoria Leaf Spot", "Tomato Late Blight",
    "Tomato Healthy", "Tomato Early Blight", "Tomato Bacterial Spot", "Strawberry Leaf Scorch",
    "Strawberry Healthy", "Potato Late Blight", "Potato Healthy", "Potato Early Blight",
    "Peach Healthy", "Peach Bacterial Spot", "Grape Leaf Blight", "Grape Healthy",
    "Grape Esca", "Grape Black Rot", "Corn Northern Leaf Blight", "Corn Healthy",
    "Corn Common Rust", "Corn Leaf Spot", "Cherry Powdery Mildew", "Cherry Healthy",
    "Bell Pepper Healthy", "Bell Pepper Bacterial Spot", "Apple Healthy", "Cedar Apple Rust",
    "Apple Black Rot", "Apple Scab"
]

def decode_prediction(prediction):
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class] if predicted_class < len(class_labels) else "Unknown Class"

# Fetch disease causes and treatments from Gemini AI
def get_disease_info(disease_name):
    prompt = f"""
    Provide a detailed explanation for the causes and treatment of {disease_name} in plants.
    Format the response as:
    **Causes:** <cause details>
    **Treatment:** <treatment details>
    """
    try:
        response = model.generate_content(prompt)
        return response.text if response and response.text else "No information available."
    except Exception as e:
        return f"Error fetching data: {str(e)}"

# Simulate real-time weather data
def get_simulated_weather():
    temperature = round(random.uniform(20, 35), 1)
    humidity = round(random.uniform(50, 80), 1)
    return temperature, humidity

# Simulate real-time soil data
def get_simulated_soil_data():
    pH = round(random.uniform(5.5, 7.0), 2)
    npk = (round(random.uniform(15, 25), 1), round(random.uniform(10, 20), 1), round(random.uniform(10, 20), 1))
    moisture = round(random.uniform(50, 70), 1)
    return pH, npk, moisture

# Predict disease based on environmental conditions
def predict_disease_based_on_conditions(temp, humidity, pH, npk, moisture):
    diseases = []
    if pH < 5.5 and moisture > 60:
        diseases.append("Tomato Late Blight")
    if humidity > 60 and 15 < temp < 30:
        diseases.append("Leaf Mold")
    if humidity > 70 and temp < 25:
        diseases.append("Tomato Target Spot")
    
    return diseases if diseases else ["No specific disease identified based on conditions."]

# Streamlit UI
st.title("🌱 AI-Based Plant Disease Detection & Prediction")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image of the affected plant...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)
    
    # Process and predict
    processed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    disease = decode_prediction(output_data)
    
    st.subheader("🦠 Disease Detection")
    st.write(f"**Detected Disease:** {disease}")

    # Fetch real causes and treatment from Gemini AI
    disease_info = get_disease_info(disease)
    st.write(disease_info)

    # Simulate real-time weather and soil data
    temperature, humidity = get_simulated_weather()
    pH, npk, moisture = get_simulated_soil_data()

    st.subheader("🌦️ Environmental Conditions")
    st.write(f"**Temperature:** {temperature}°C, **Humidity:** {humidity}%")
    
    st.subheader("🌱 Soil Conditions")
    st.write(f"**pH:** {pH}, **NPK:** {npk}, **Moisture:** {moisture}%")

    # Predict diseases based on conditions
    possible_diseases = predict_disease_based_on_conditions(temperature, humidity, pH, npk, moisture)
    
    st.subheader("🔮 AI-based Disease Prediction")
    st.write(f"**Possible Diseases Based on Conditions:** {', '.join(possible_diseases)}")

    for disease in possible_diseases:
        disease_info = get_disease_info(disease)
        st.write(f"🔍 **{disease}**")
        st.write(disease_info)

    # Additional farming suggestions
    st.subheader("💡 Farming Recommendations")
    st.write("🌾 Consider crop rotation with resistant plants like spinach or lettuce to minimize future risks.")
    st.write("💧 Ensure proper ventilation, avoid overwatering, and use organic fertilizers for soil enrichment.")

    # Chatbot Section
    st.subheader("💬 Ask the AI about the Detected Disease")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    user_input = st.text_input("Type your question here...", key="user_question")

    if user_input:
        chatbot_prompt = f"""
        You are an AI assistant that helps users understand plant diseases.
        The user has detected a disease: **{disease}**.
        Answer their questions based on this disease.
        
        User: {user_input}
        AI:
        """
        try:
            response = model.generate_content(chatbot_prompt)
            reply = response.text if response and response.text else "Sorry, I couldn't find an answer."

            # Store conversation history
            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append(("AI", reply))

        except Exception as e:
            reply = f"Error fetching response: {str(e)}"

        st.write(reply)

    # Show chat history
    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for sender, message in st.session_state.chat_history:
            st.write(f"**{sender}:** {message}")
