import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import google.generativeai as genai
import random

st.set_page_config(page_title="AI-Based Plant Disease Detector")

genai.configure(api_key="AIzaSyBTbpL_tjWCTl6ZDKlHUNFIYwk0irVAfac")
model = genai.GenerativeModel("gemini-1.5-pro")

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path='model_unquant.tflite')
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

disease_labels = [
    "Tomato Yellow Leaf Curl Virus", "Tomato Septoria Leaf Spot", "Tomato Late Blight",
    "Tomato Healthy", "Tomato Early Blight", "Tomato Bacterial Spot",
    "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Potato Late Blight", "Potato Healthy", "Potato Early Blight",
    "Peach Healthy", "Peach Bacterial Spot",
    "Grape Leaf Blight", "Grape Healthy", "Grape Esca", "Grape Black Rot",
    "Corn Northern Leaf Blight", "Corn Healthy", "Corn Common Rust", "Corn Leaf Spot",
    "Cherry Powdery Mildew", "Cherry Healthy",
    "Bell Pepper Healthy", "Bell Pepper Bacterial Spot",
    "Apple Healthy", "Cedar Apple Rust", "Apple Black Rot", "Apple Scab"
]

def generate_environment_data():
    return {
        "temperature": round(random.uniform(15, 35), 2),
        "humidity": round(random.uniform(30, 90), 2),
        "soil_pH": round(random.uniform(5.0, 7.5), 2),
        "soil_moisture": round(random.uniform(10, 50), 2)
    }

def predict_disease(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    
    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']
    
    interpreter.set_tensor(input_tensor_index, image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_tensor_index)
    
    predicted_label_index = np.argmax(output)
    return disease_labels[predicted_label_index]

def get_disease_info(disease_name, env_data):
    prompt = (f"What are the causes and treatments for {disease_name} in plants? "
              f"How do the following conditions affect it?\n"
              f"Temperature: {env_data['temperature']}°C, "
              f"Humidity: {env_data['humidity']}%, "
              f"Soil pH: {env_data['soil_pH']}, "
              f"Soil Moisture: {env_data['soil_moisture']}%.")
    response = model.generate_content(prompt)
    return response.text if response else "No information found."

st.title("🌿 AI-Based Plant Disease Detector")

st.sidebar.title("📌 How to Use")
st.sidebar.write("1️⃣ Upload an image of a leaf.")  
st.sidebar.write("2️⃣ The AI will predict the disease.")  
st.sidebar.write("3️⃣ View the cause and treatment.")  
st.sidebar.write("4️⃣ Environmental data is simulated and included.")

st.sidebar.title("🌾 Supported Classes")
for disease in disease_labels:
    st.sidebar.write(f"✅ {disease}")

uploaded_file = st.file_uploader("📤 Upload an image of a plant leaf", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    display_image = image.resize((image.width // 2, image.height // 2))
    st.image(display_image, caption="📷 Uploaded Image (Reduced Display Size)", use_column_width=True)
    
    env_data = generate_environment_data()
    
    st.subheader("🌡 Simulated Environmental Data")
    st.write(f"- **Temperature:** {env_data['temperature']}°C")
    st.write(f"- **Humidity:** {env_data['humidity']}%")
    st.write(f"- **Soil pH:** {env_data['soil_pH']}")
    st.write(f"- **Soil Moisture:** {env_data['soil_moisture']}%")
    
    with st.spinner("Analyzing..."):
        predicted_disease = predict_disease(image)
    
    st.success(f"✅ **Detected Disease:** {predicted_disease}")
    
    with st.spinner("Fetching treatment details..."):
        disease_info = get_disease_info(predicted_disease, env_data)
    
    st.subheader("📖 Disease Information")
    st.write(disease_info)
    
    st.subheader("💬 Ask AI About This Disease")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    user_query = st.text_input("Ask a question about this disease:")
    if st.button("Submit Query") and user_query:
        with st.spinner("Generating AI response..."):
            chat_prompt = (f"{user_query}\n\nDetected Disease: {predicted_disease}\n"
                           f"Environmental Data: {env_data}")
            ai_response = model.generate_content(chat_prompt).text
            st.session_state.chat_history.append((user_query, ai_response))
    
    for query, response in st.session_state.chat_history:
        st.write(f"**You:** {query}")
        st.write(f"**AI:** {response}")
