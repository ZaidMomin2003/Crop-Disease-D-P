import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import google.generativeai as genai

# ✅ Set page configuration
st.set_page_config(page_title="AI-Based Plant Disease Detector", layout="wide")

# ✅ Configure Gemini AI
genai.configure(api_key="AIzaSyBoFhFSFz7VfEo5QdgGpyhixL-vRGq-3Qc")
model = genai.GenerativeModel("gemini-pro")

# ✅ Load the TensorFlow Lite Model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path='model_unquant.tflite')
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# ✅ Define Updated Disease Labels (28 Classes)
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

# ✅ Function to Make Predictions
def predict_disease(image):
    image = image.resize((224, 224))  # ✅ Fix: Ensure correct input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0).astype(np.float32)

    input_tensor_index = interpreter.get_input_details()[0]['index']
    output_tensor_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_tensor_index, image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_tensor_index)

    predicted_label_index = np.argmax(output)
    return disease_labels[predicted_label_index]

# ✅ Function to Fetch AI-based Cause & Treatment
def get_disease_info(disease_name):
    prompt = f"What are the causes and treatments for {disease_name} in plants?"
    response = model.generate_content(prompt)
    return response.text if response else "No information found."

# ✅ Function for Chatbot Response
def chatbot_response(user_question, disease_name):
    prompt = f"The detected disease is {disease_name}. Answer this user query: {user_question}"
    response = model.generate_content(prompt)
    return response.text if response else "I'm not sure. Please consult an expert."

# ✅ Streamlit UI
st.title("🌿 AI-Based Plant Disease Detector")

# ✅ Sidebar: Instructions + Updated Supported Crops
st.sidebar.title("📌 How to Use")
st.sidebar.write("1️⃣ Upload an image of a leaf.")  
st.sidebar.write("2️⃣ The AI will predict the disease.")  
st.sidebar.write("3️⃣ View the cause and treatment.")  
st.sidebar.write("4️⃣ Chat with AI for more help.")  

st.sidebar.title("🌾 Supported Classes")
for disease in disease_labels:
    st.sidebar.write(f"✅ {disease}")

# ✅ File Upload
uploaded_file = st.file_uploader("📤 Upload an image of a plant leaf", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Decrease **display** size, but keep correct **model input size**
    display_image = image.resize((image.width // 2, image.height // 2))

    st.image(display_image, caption="📷 Uploaded Image (Reduced Display Size)", use_column_width=True)

    # 🔍 Make Prediction
    with st.spinner("Analyzing..."):
        predicted_disease = predict_disease(image)
    
    st.success(f"✅ **Detected Disease:** {predicted_disease}")

    # 🔎 Fetch Cause & Treatment from Gemini AI
    with st.spinner("Fetching treatment details..."):
        disease_info = get_disease_info(predicted_disease)

    st.subheader("📖 Disease Information")
    st.write(disease_info)

    # 🤖 Chatbot Section
    st.subheader("💬 Ask AI About This Disease")
    user_question = st.text_input("Ask a question about this disease:")
    
    if user_question:
        with st.spinner("Thinking..."):
            chatbot_reply = chatbot_response(user_question, predicted_disease)
        
        st.write(f"**AI:** {chatbot_reply}")
