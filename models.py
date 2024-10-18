from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid
db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'User'
    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String, nullable=False)
    username = db.Column(db.String, unique=True, nullable=False)
    email = db.Column(db.String, unique=True, nullable=True)
    password = db.Column(db.String, nullable=False)

    # Relationships
    students = db.relationship('Student', backref='user', lazy=True)
    
class Student(db.Model):
    __tablename__ = 'Student'
    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    firstname = db.Column(db.String, nullable=False)
    lastname = db.Column(db.String, nullable=False)
    user_id = db.Column(db.String, db.ForeignKey('User.id'), nullable=False)
    sex = db.Column(db.String)  # Consider using Enum
    grade = db.Column(db.String)  # Consider using Enum

    # Relationships
    voice_exercises_history = db.relationship('VoiceExcercisesHistory', backref='student', lazy=True)
    
class VoiceExcercises(db.Model):
    __tablename__ = 'VoiceExcercises'  

    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = db.Column(db.String)
    voiceImage = db.Column(db.Text)
    voice = db.Column(db.String, nullable=False)
    grade = db.Column(db.Enum('GradeOne', 'GradeTwo', 'GradeThree', name='grade_enum'))
    createdAt = db.Column(db.DateTime, server_default=db.func.now())
    updatedAt = db.Column(db.DateTime, onupdate=db.func.now())
   
    moduleId = db.Column(db.String, db.ForeignKey('Module.id'), nullable=False)  

    history = db.relationship('VoiceExcercisesHistory', backref='voice_excercises')
    

class VoiceExcercisesHistory(db.Model):
    __tablename__ = 'VoiceExcercisesHistory'
    
    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    voice = db.Column(db.String)
    voiceImage = db.Column(db.String)
    voiceRecord = db.Column(db.Text)
    recognizedText = db.Column(db.String)
    accuracyScore = db.Column(db.Float)
    pronunciationScore = db.Column(db.Float)
    fluencyScore = db.Column(db.Float)
    speedScore = db.Column(db.Float)
    phonemes = db.Column(db.String)
    createdAt = db.Column(db.DateTime, default=datetime.utcnow)
    voiceExercisesId = db.Column(db.String, db.ForeignKey('VoiceExcercises.id')) 
    studentId = db.Column(db.String, db.ForeignKey('Student.id')) 
    score = db.Column(db.Integer) 

   
class Module(db.Model):
    __tablename__ = 'Module'  

    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    moduleTitle = db.Column(db.String, nullable=False)
    moduleDescription = db.Column(db.String)
    learnOutcome1 = db.Column(db.String)
    videoModule = db.Column(db.String)
    imageModule = db.Column(db.String)

   
    subjects = db.Column(db.Enum('Reading', 'Math',  name='subject_enum'))

    createdAt = db.Column(db.DateTime, server_default=db.func.now())
    updatedAt = db.Column(db.DateTime, onupdate=db.func.now())


    VoiceExcercises = db.relationship('VoiceExcercises', backref='module')  