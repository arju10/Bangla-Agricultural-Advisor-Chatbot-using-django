import json
from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import tensorflow as tf
import os
from django.core.files.storage import FileSystemStorage
import re
from django.db import transaction
import threading

import uuid  # For generating unique file names

def sanitize_filename(filename):
    """Sanitize the file name by removing spaces and special characters."""
    name, ext = os.path.splitext(filename)  # Split into name and extension
    sanitized_name = re.sub(r'[^\w\-\.]', '_', name)  # Replace special characters with underscores
    return f"{sanitized_name}{ext}"

# Data Preprocessing
def load_data():
    df = pd.read_csv("/bangla_agriculture_chatbot/data/agriculture_data.csv")
    return df

# Load dataset globally
data = load_data()

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['Question (Bangla)'])

# Save TF-IDF model for later use (optional, if not already saved)
joblib.dump(vectorizer, "/bangla_agriculture_chatbot/chatbot/static/models/tfidf_vectorizer.joblib")
joblib.dump(tfidf_matrix, "/bangla_agriculture_chatbot/chatbot/static/models/tfidf_matrix.joblib")

def get_best_match(query):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    best_match_index = similarities.argmax()
    return data.iloc[best_match_index]['Answer (Bangla)']

# Load Disease Detection Model
disease_model = tf.keras.models.load_model("/bangla_agriculture_chatbot/chatbot/static/models/disease_detection_model.h5")

# Load Class Labels
with open("/bangla_agriculture_chatbot/chatbot/static/models/class_labels.json", "r") as f:
    class_labels = json.load(f)

# # Detect Disease 
def detect_disease(image_path):
    print("detect_disease function called!")  # Debugging statement
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    # Predict the class
    prediction = disease_model.predict(img_array)
    predicted_class_index = prediction.argmax()
    predicted_class = class_labels[predicted_class_index]

    print(f"Predicted Class Index: {predicted_class_index}")
    print(f"Predicted Class (Bangla): {predicted_class}")

    # Clean up the predicted class name (remove punctuation and convert to lowercase)
    cleaned_predicted_class = re.sub(r'[^\w\s]', '', predicted_class).lower()
    print(f"Cleaned Predicted Class: {cleaned_predicted_class}")

    # Find the suggestion for the predicted disease
    disease_row = None
    for _, row in data.iterrows():
        if row['Category'] == 'Disease Control':
            # Clean up the question (remove punctuation and convert to lowercase)
            cleaned_question = re.sub(r'[^\w\s]', '', row['Question (Bangla)']).lower()
            print(f"Checking against: {cleaned_question}")  # Debugging statement
            if cleaned_predicted_class in cleaned_question:
                disease_row = row
                print(f"Match found: {row['Question (Bangla)']}")  # Debugging statement
                break

    if disease_row is not None:
        bangla_disease_name = disease_row['Question (Bangla)']
        suggestion = disease_row['Answer (Bangla)']
        return f"রোগ: {bangla_disease_name}\nসাজেশন: {suggestion}"
    else:
        return "রোগ শনাক্ত করা যায়নি। দয়া করে একটি অন্য ছবি আপলোড করুন।"

#   # Speech Recognition Function
import speech_recognition as sr

def speech_to_text():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Listening... Please speak your query.")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            # Recognize speech using Google Web Speech API
            query = recognizer.recognize_google(audio, language="bn-BD")  # Use Bangla language
            print(f"User Query: {query}")
            return query.strip()
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None


from django.conf import settings

def chatbot_view(request):
    if request.method == 'POST':
        # 1️⃣ 📷 Handle SAMPLE IMAGE request (sent as JSON)
        if request.content_type == 'application/json':
            body_unicode = request.body.decode('utf-8')
            body_data = json.loads(body_unicode)
            sample_image = body_data.get('sample_image')

            if sample_image:
                static_sample_path = os.path.join(
                    settings.BASE_DIR,
                    'chatbot',
                    'static',
                    'samples',
                    sample_image
                )
                static_sample_url = f"/static/samples/{sample_image}"

                print(f"Sample image path: {static_sample_path}")

                disease_result = detect_disease(static_sample_path)

                response_data = {
                    'image_url': static_sample_url,
                    'disease_result': disease_result
                }

                return JsonResponse(response_data, json_dumps_params={'ensure_ascii': False})

        # 2️⃣ Handle text query
        user_query = request.POST.get('user_query', '').strip()
        if user_query:
            answer = get_best_match(user_query)
            return render(request, 'chatbot/index.html', {'answer': answer})

        # 3️⃣ Handle voice query
        if request.POST.get('voice_query') == 'true':
            user_query = speech_to_text()  # Convert speech to text
            print(f"Recognized Query: {user_query}")  # Debugging statement
            if user_query:
                answer = get_best_match(user_query)  # Get the chatbot's response
                print(f"Passing Query to Template: {user_query}")  # Debugging statement
                return render(request, 'chatbot/index.html', {'answer': answer, 'query': user_query})
            else:
                return render(request, 'chatbot/index.html', {'answer': "অনুগ্রহ করে আবার বলুন।", 'query': "কিছুই শোনা যায়নি।"})

        # 4️⃣ Handle image upload from local computer
        uploaded_image = request.FILES.get('image_upload')
        if uploaded_image:
            fs = FileSystemStorage()
            try:
                original_filename = uploaded_image.name
                sanitized_filename = sanitize_filename(original_filename)
                unique_filename = f"{uuid.uuid4().hex}_{sanitized_filename}"

                filename = fs.save(unique_filename, uploaded_image)
                image_url = fs.url(filename)
                image_path = fs.path(filename)

                print(f"Uploaded Image Path: {image_path}")

                disease_result = detect_disease(image_path)
                print(f"Disease Detection Result: {disease_result}")

                def cleanup_file():
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"File removed: {image_path}")

                timer = threading.Timer(10.0, cleanup_file)
                timer.start()

                response_data = {
                    'image_url': image_url,
                    'disease_result': disease_result
                }
                print(f"JSON Response: {response_data}")
                return JsonResponse(response_data, json_dumps_params={'ensure_ascii': False})

            except Exception as e:
                print(f"Error during image processing: {str(e)}")
                return JsonResponse({'error': f'ছবি প্রক্রিয়াকরণে সমস্যা হয়েছে: {str(e)}'}, status=500)

    # Default GET or fallback render
    return render(request, 'chatbot/index.html')
