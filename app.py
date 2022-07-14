
from datetime import datetime
import pytz
from flask import Flask , jsonify , request
import numpy as np
import scipy
from scipy.signal import filtfilt
from scipy import stats
from sklearn import preprocessing
import tensorflow as tf
import json
from json import JSONEncoder
from firebase import firebase







app = Flask(__name__)


    
    
def pandpassfilter(signal):

  fs = 125
  Low_freuency = 0.5
  High_frequency = 50
  nyq = 0.5 * fs
  Low = Low_freuency / nyq
  High  = High_frequency / nyq
  order = 2

  b , a = scipy.signal.butter(order , [Low , High] , 'bandpass' , analog=False)

  y = scipy.signal.filtfilt( b , a , signal , axis = 0 )

  return (y)

def signal_preprocessing(sensor_data):
    sensor_data = np.array(sensor_data)
    sensor_data = sensor_data[:1250]
    sensor_data = sensor_data.reshape(-1,1)
    filterd_signal = pandpassfilter(sensor_data)
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_signal = min_max_scaler.fit_transform(filterd_signal)
    peaks,_ = scipy.signal.find_peaks(normalized_signal[: , 0] , distance=50)
    for i in range(0 , len(normalized_signal)):

        if normalized_signal[i] > 0.8:

            start = peaks[4]
            end = peaks[5]

            end  = int(end*1.1)

    signal = normalized_signal[ start:end, : ]        
    padding = (np.zeros(186  - len(signal)))   
    signal = np.append(signal , padding)  

    return signal


 
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


#ECG predict method

@app.route('/' , methods = ['POST'])

def predict():
    
    firebase1 = firebase.FirebaseApplication('https://wireless-ecg-esp32-default-rtdb.firebaseio.com/', None)
    result = firebase1.get('test/', '')

    signal = []

    for i in result['int']:
       signal.append(i)
      
    saved_model_path = './ECG97_Model.h5'
    another_strategy = tf.distribute.MirroredStrategy()
    with another_strategy.scope():
        load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
        ECG_model = tf.keras.models.load_model(saved_model_path, options=load_options)
    # int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    # features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model    
    # signal = pd.read_csv('test.txt' , delimiter='\t') 
    signal = signal_preprocessing(signal)
    signal =  np.array(signal)
    signal = signal.reshape(1 , -1)
    prediction = ECG_model.predict(signal)
    prediction = np.array(prediction)
    prediction = prediction.ravel()
    max_pred = max(prediction)
    numpyData = {"signal": signal}
    array = json.dumps(numpyData, cls=NumpyArrayEncoder)

    if max_pred == prediction[0]:

        return jsonify({'signal':array , 'prediction':'normal', 'precentage':prediction[0]*100,'date': datetime.now(pytz.timezone('Africa/Cairo'))})

    elif max_pred == prediction[1]:
        prediction =  jsonify({'signal':array,'prediction':'Supra-ventricular premature','precentage':prediction[1] * 100, 'date': datetime.now(pytz.timezone('Africa/Cairo'))})

    elif max_pred == prediction[2]:
        prediction =  jsonify({'signal':array,'prediction':'Premature ventricular contraction','precentage':prediction[2] * 100, 'date':datetime.now(pytz.timezone('Africa/Cairo'))})

    elif max_pred == prediction[3]:

        prediction =  jsonify({'signal':array,'prediction':'Fusion of ventricular and normal','precentage':prediction[3] * 100, 'date': datetime.now(pytz.timezone('Africa/Cairo'))})

    elif max_pred == prediction[4]:
        prediction =  jsonify({'signal':array,'prediction':'Unclassifiable beat','precentage':prediction[4]* 100, 'date': datetime.now(pytz.timezone('Africa/Cairo'))})

          
    return prediction


if __name__ == '__main__':
    app.run(debug=True , port = 8000)





