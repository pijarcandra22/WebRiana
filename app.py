from flask import Flask, render_template, request, url_for, redirect,session
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
import tensorflow as tf
from tensorflow import keras
import librosa
import librosa.display
import numpy as np
import uuid
import os
import pywt
import pandas as pd

app = Flask(__name__)
app = Flask(__name__,template_folder='temp')

scaler = StandardScaler()
scaler.fit(pd.read_csv("features_mfcc&dwt.csv").iloc[:,:-1].values)
model = keras.models.load_model("py/model.h5")

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_audio(directory,name_file):
  if name_file not in request.files:
    return False
  file = request.files[name_file]
  if file.filename == '':
    return False
  if file and allowed_file(file.filename):
    app.config['UPLOAD_FOLDER'] = directory
    filename = secure_filename(file.filename)
    formatfile=filename.split('.')
    newfilename=str(uuid.uuid4().hex)+'.'+formatfile[1]
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    os.rename(os.path.join(app.config['UPLOAD_FOLDER'], filename),os.path.join(app.config['UPLOAD_FOLDER'], newfilename))
    return newfilename

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
  audio=upload_audio("data","audio")
  if not audio:
    return "None"
  
  feature = get_features("data/"+audio)
  X = []
  for ele in feature:
    X.append(ele)
  
  X = np.array(X)
  X = scaler.transform(X)
  predict = model.predict(X)
  bigpredict = [max(x) for x in predict]
  bestpredict = predict[bigpredict.index(max(bigpredict))].tolist()
  hasil = bestpredict.index(max(bestpredict))
  return str(hasil)

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def extract_features(data,sample_rate):
    result = np.array([])

    cA, cD = pywt.dwt(data,'bior6.8','per');
    
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=cA, sr=sample_rate,n_mfcc=200).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    mfcc = np.mean(librosa.feature.mfcc(y=cD, sr=sample_rate,n_mfcc=200).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    
    return result

def get_features(path):
    data, sample_rate = librosa.load(path, duration=30, offset=0.6,sr=16000)
    
    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

if __name__=='__main__':
  app.run()