<!-- # Project Setup
### Step-1: Create `venv` (virtual machine)
```bash
python3 -m venv venv
```

### Step-2: Activate `venv`
```bash
source venv/bin/activate
```
### Step-3: Install All Dependencies

```bash
pip install -r requirements.txt
```

### Step-4: Run the project
```bash
python manage.py runserver
``` -->

# Bangla Agricultural Advisor Chatbot

## Overview

The **Bangla Agricultural Advisor Chatbot** is a Django-based web application designed to assist farmers and agricultural enthusiasts in Bangladesh by providing advice and solutions in the Bengali language. The chatbot supports three interaction modes:

1. **Text Query**: Users can type their questions in Bengali and receive relevant advice.
2. **Voice Query**: Users can speak their queries in Bengali, and the chatbot will transcribe and respond with advice.
3. **Image Upload**: Users can upload images of crops to detect diseases and receive actionable suggestions.

The project leverages machine learning models for natural language processing (NLP) and image classification to provide accurate responses.

---

## Features

### 1. Text-Based Query System
- Users can input their questions in Bengali.
- The system uses a TF-IDF vectorizer and cosine similarity to match user queries with pre-defined answers from a dataset.

### 2. Voice Recognition
- Users can speak their queries in Bengali using a microphone.
- The system uses Google Web Speech API to transcribe the audio into text and processes it similarly to text queries.

### 3. Disease Detection via Image Upload
- Users can upload images of crops to detect diseases.
- A TensorFlow-based deep learning model classifies the uploaded image and matches it with relevant disease control advice from the dataset.

### 4. Dynamic User Interface
- The frontend dynamically updates based on user interactions (text, voice, or image).
- Separate sections are displayed for disease detection results and advice.

---

## Project Structure

```
bangla_agriculture_chatbot/
├── chatbot/
│   ├── static/
│   │   ├── models/               # Pre-trained ML models and vectorizers
│   │   │   ├── disease_detection_model.h5
│   │   │   ├── tfidf_vectorizer.joblib
│   │   │   ├── tfidf_matrix.joblib
│   │   │   └── class_labels.json
│   ├── templates/
│   │   └── chatbot/
│   │       └── index.html        # Frontend HTML template
│   ├── views.py                  # Django view logic
│   └── urls.py                   # URL routing
├── data/
│   └── agriculture_data.csv      # Dataset containing questions and answers
├── manage.py                     # Django management script
└── README.md                     # Project documentation
```

---

## Installation and Setup

<!-- ### Prerequisites
- Python 3.8+
- Django 4.x
- TensorFlow 2.x
- Pandas, Scikit-learn, Joblib
- SpeechRecognition library
- Google Web Speech API (for voice recognition) -->

### Steps to Set Up

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/bangla-agricultural-advisor-chatbot.git
   cd bangla-agricultural-advisor-chatbot
   ```
2. **Create `venv` (virtual machine)**
    ```bash
    python3 -m venv venv
    ```

3. **Activate `venv`**
    ```bash
    source venv/bin/activate
    ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download Pre-trained Models**
   - Place the pre-trained models (`disease_detection_model.h5`, `tfidf_vectorizer.joblib`, `tfidf_matrix.joblib`, and `class_labels.json`) in the `chatbot/static/models/` directory.

6. **Prepare Dataset**
   - Place the dataset file (`agriculture_data.csv`) in the `data/` directory.

7. **Run Migrations**
   ```bash
   python manage.py migrate
   ```

8. **Start the Development Server**
   ```bash
   python manage.py runserver
   ```

9. **Access the Application**
   Open your browser and navigate to `http://127.0.0.1:8000`.

---

## Usage

### 1. Text Query
- Enter your question in the text input field and click "পরামর্শ নিন".
- The chatbot will display the most relevant advice based on your query.

### 2. Voice Query
- Click the microphone button ("🎤 কথা বলুন").
- Speak your query in Bengali.
- The chatbot will transcribe your speech and provide advice.

### 3. Image Upload
- Click "ফসলের ছবি আপলোড করুন" to upload an image of a crop.
- The chatbot will analyze the image, detect any diseases, and provide actionable suggestions.

---

## Backend Logic

### 1. Text Matching
- The system uses a TF-IDF vectorizer to transform user queries and pre-defined questions into numerical vectors.
- Cosine similarity is used to find the best match between the user query and the dataset.

### 2. Disease Detection
- The uploaded image is preprocessed and fed into a TensorFlow-based deep learning model.
- The model predicts the disease class, which is then matched with relevant advice from the dataset.

### 3. Voice Recognition
- Audio input is captured using the microphone and transcribed into text using Google Web Speech API.
- The transcribed text is processed similarly to text queries.

---

## Dataset

The dataset (`agriculture_data.csv`) contains three columns:
- **Category (Bangla)**: Questions category.
- **Question (Bangla)**: Common agricultural questions in Bengali.
- **Answer (Bangla)**: Corresponding advice or solutions in Bengali.

Ensure that the dataset is updated regularly to improve the chatbot's accuracy.



<!-- ## Deployment -->




## Future Enhancements

1. **Multilingual Support**: Add support for other regional languages.
2. **Improved Disease Detection**: Fine-tune the deep learning model for better accuracy.
3. **Mobile App**: Develop a mobile-friendly version of the chatbot.
4. **User Authentication**: Allow users to save their queries and track previous interactions.





## Contact

For any questions or feedback, please contact:
- Email: mst.tahminajerinarju@example.com
- GitHub: [@arju10](https://github.com/arju10)
- LinkedIn: [@arju10](https://www.linkedin.com/in/arju10/)
- Facebook: [@arju10.arju](https://www.facebook.com/arju10.arju)
---
# Bangla-Agricultural-Advisor-Chatbot-using-django
