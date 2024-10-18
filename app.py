from flask import Flask, jsonify, request
from models import db, VoiceExcercises, VoiceExcercisesHistory, Module
import os
from dotenv import load_dotenv
from flask_cors import CORS
import speech_recognition as sr
from pydub import AudioSegment
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import torch.optim as optim
from jiwer import wer
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
import nltk
from nltk.corpus import cmudict
from pydub.exceptions import CouldntDecodeError
from difflib import SequenceMatcher
import json
import base64
from flask_migrate import Migrate
load_dotenv()
migrate = Migrate()
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
migrate.init_app(app, db) 

CORS(app, origins=["https://capstone-lms-red.vercel.app"], 
     methods=["GET", "POST", "PUT", "DELETE"], 
     allow_headers=["Content-Type", "Authorization"], 
     supports_credentials=True)

@app.route('/api/voice-exercises', methods=['GET'])
def get_voice_exercises():
    try:
        # Get the module title from query parameters
        module_title = request.args.get('moduleTitle')

        if module_title:
            # Filter by module title if provided
            exercises = VoiceExcercises.query.join(Module).filter(Module.moduleTitle == module_title).all()
        else:
            # Query all records if no module title is provided
            exercises = VoiceExcercises.query.all()

        results = [
            {
                'id': ex.id,
                'voice': ex.voice,
                'grade': ex.grade,
                'voiceImage': ex.voiceImage,
                'userId': ex.userId,
                'createdAt': ex.createdAt,
                'updatedAt': ex.updatedAt,
            }
            for ex in exercises
        ]
        return jsonify(results), 200

    except Exception as e:
        print(f"Error fetching voice exercises: {e}")
        return jsonify({'error': 'An error occurred while fetching data.'}), 500

sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize BERT and RoBERTa models and tokenizers
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

# Initialize NLTK CMU Pronouncing Dictionary
nltk.download('cmudict')
phoneme_dict = cmudict.dict()

# Define Fluency Model
class FluencyModel(nn.Module):
    def __init__(self):
        super(FluencyModel, self).__init__()
        self.fc = nn.Linear(768 * 2, 1)  # Concatenate two BERT embeddings (768*2 dimensions)
        self.sigmoid = nn.Sigmoid()  # Use sigmoid to scale output between 0 and 1
    
    def forward(self, x):
        return self.sigmoid(self.fc(x))

# Initialize the fluency model, criterion, and optimizer
fluency_model = FluencyModel()
fluency_criterion = nn.MSELoss()
fluency_optimizer = optim.Adam(fluency_model.parameters(), lr=0.001)

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def get_roberta_embeddings(text):
    if not text:
        raise ValueError("Input text cannot be empty")
    
    inputs = roberta_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def calculate_advanced_similarity(text1, text2):
    if not text1 or not text2:
        raise ValueError("Both input texts must be non-empty")
    
    embedding1 = get_roberta_embeddings(text1)
    embedding2 = get_roberta_embeddings(text2)
    
    cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
    
    return cosine_sim.item() * 100  # Convert to percentage

def calculate_accuracy(expected_text, recognized_text):
    if not expected_text or not recognized_text:
        return 0.0

    wer_score = wer(expected_text, recognized_text) * 100  # Convert to percentage
    semantic_similarity = calculate_advanced_similarity(expected_text, recognized_text)

    combined_accuracy = (100 - wer_score + semantic_similarity) / 2
    return round(combined_accuracy, 2)

def calculate_semantic_similarity(text1, text2):
    if not text1 or not text2:
     return 0.0
    inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs1 = bert_model(**inputs1)
        outputs2 = bert_model(**inputs2)
    embedding1 = outputs1.last_hidden_state.mean(dim=1).squeeze()
    embedding2 = outputs2.last_hidden_state.mean(dim=1).squeeze()
    cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
    return cosine_sim.item() * 100  # Convert to percentage

def calculate_fluency_score(expected_text, recognized_text):
    if not recognized_text or not expected_text:
        return 0.0

    semantic_similarity = calculate_semantic_similarity(expected_text, recognized_text)

    matcher = SequenceMatcher(None, expected_text.lower(), recognized_text.lower())
    word_match_ratio = matcher.ratio() * 100

    similarity_threshold = 10
    if semantic_similarity < similarity_threshold and word_match_ratio < similarity_threshold:
        return 0.0

    fluency_score = (semantic_similarity + word_match_ratio) / 2
    return round(fluency_score, 2)

def calculate_speed_score(audio_file, recognized_text):
    audio = AudioSegment.from_file(audio_file)
    duration_seconds = len(audio) / 1000  # seconds
    num_words = len(recognized_text.split())

    if duration_seconds > 0:
        wps = num_words / duration_seconds
    else:
        wps = 0

    # Speed score based on WPM
    if wps < 0.01:
        return 0
    elif 0.01 <= wps < 0.02:
        return 5
    elif 0.02 <= wps < 0.05:
        return 10
    elif 0.05 <= wps < 0.1:
        return 15
    elif 0.1 <= wps < 0.15:
        return 20
    elif 0.15 <= wps < 0.2:
        return 25
    elif 0.2 <= wps < 0.25:
        return 30
    elif 0.25 <= wps < 0.3:
        return 35
    elif 0.3 <= wps < 0.4:
        return 40
    elif 0.4 <= wps < 0.5:
        return 45
    elif 0.5 <= wps < 0.6:
        return 50
    elif 0.6 <= wps < 0.75:
        return 55
    elif 0.75 <= wps < 1.0:
        return 60
    elif 1.0 <= wps < 1.25:
        return 70
    elif 1.25 <= wps < 1.5:
        return 80
    elif 1.5 <= wps < 1.75:
        return 85
    elif 1.75 <= wps < 2.0:
        return 90
    elif 2.0 <= wps < 2.25:
        return 92
    elif 2.25 <= wps < 2.5:
        return 95
    elif 2.5 <= wps < 2.75:
        return 98
    elif 2.75 <= wps < 3.0:
        return 99
    else:
        return 100

def get_phonemes(text):
    words = text.lower().split()
    phoneme_list = []
    for word in words:
        if word in phoneme_dict:
            phoneme_list.append(phoneme_dict[word][0])
        else:
            phoneme_list.append("No phonemes available")  
    return phoneme_list

def recognize_speech(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    
def get_grade(final_score):
    if final_score >= 90:
        return 'Excellent'
    elif final_score >= 85:
        return 'Very Satisfactory'
    elif final_score >= 80:
        return 'Satisfactory'
    elif final_score >= 75:
        return 'Fairly Satisfactory'
    else:
        return 'Did Not Meet Expectations'

@app.route('/api/voice-exercises-history', methods=['POST'])
def create_voice_exercise_history():
    try:
        if 'audio_blob' in request.files and 'expected_text' in request.form:
            audio_blob = request.files['audio_blob']
            expected_text = request.form['expected_text']
            audio_filename = 'recorded_audio.webm'
            audio_path = os.path.join(app.root_path, 'static', audio_filename)

            # Save uploaded file
            audio_blob.save(audio_path)

            # Convert WebM to WAV using PyDub
            audio = AudioSegment.from_file(audio_path, format="webm")
            wav_path = os.path.join(app.root_path, 'static', 'recorded_audio.wav')
            audio.export(wav_path, format="wav")

            try:
                recognized_text = recognize_speech(wav_path)
            except sr.UnknownValueError:
                recognized_text = ""
            except sr.RequestError:
                return jsonify({'error': 'Could not process audio; please try again'})

            # Calculate scores
            accuracy_score = calculate_accuracy(expected_text, recognized_text)
            pronunciation_score = calculate_semantic_similarity(expected_text, recognized_text)
            fluency_score = calculate_fluency_score(expected_text, recognized_text)
            speed_score = calculate_speed_score(wav_path, recognized_text)

            final_score = (accuracy_score + fluency_score + pronunciation_score + speed_score) / 4
            final_score = round(final_score, 2)
            grade = get_grade(final_score)

            phonemes = json.dumps(get_phonemes(recognized_text)) 
            voice_image = request.form.get('voice_image')
            voice_exercises_id = request.form.get('voice_exercises_id')
            student_id = request.form.get('student_id')

            if not student_id or not voice_exercises_id:
                return jsonify({'error': 'student_id and voice_exercises_id are required'}), 400
            
            with open(wav_path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

            # Save the data to the database
            new_history = VoiceExcercisesHistory(
                voice=expected_text,
                voiceRecord=audio_base64,
                voiceImage=voice_image,
                recognizedText=recognized_text,
                accuracyScore=str(accuracy_score),
                pronunciationScore=str(pronunciation_score),
                fluencyScore=str(fluency_score),
                speedScore=str(speed_score),
                phonemes=phonemes,  
                voiceExercisesId=voice_exercises_id, 
                studentId=student_id,
                score=final_score   
            )
            print(f"Inserting: {new_history}")

            db.session.add(new_history)
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback() 
                app.logger.error(f"Database insertion error: {e}")
                return jsonify({'error': 'Failed to save to database'}), 500

            return jsonify({
                'accuracy_score': accuracy_score,
                'pronunciation_score': pronunciation_score,
                'fluency_score': fluency_score,
                'speed_score': speed_score,
                'final_score': final_score,
                'grade': grade,
                'recognized_text': recognized_text,
                'phonemes': phonemes
                
            }), 200  
        else:
            return jsonify({'error': 'No audio file or expected text received'}), 400
    except Exception as e:
        app.logger.error(f"Error creating voice exercise history: {e}")
        return jsonify({'error': 'An error occurred while creating history.'}), 500



if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
    app.run(debug=True)