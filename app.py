import numpy as np 
from flask import Flask, request, render_template
import joblib
from PIL import Image 

import io


app= Flask(__name__) 

try: 
    model= joblib.load('model.pkl')
    print("Model Loaded.")
except FileNotFoundError:
    print("Model loading unsucessful")



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def predict():
    if model is None:
        return render_template("index.html","Model not loaded")
    
    if 'file' not in request.files: 
        return render_tempalte("index.html","No file selected")
    

    file=request.files['file']
    if file:
        try: 
            image_bytes=file.read()
            img=Image.open(io.BytesIO(image_bytes))
            ## Pre process 

            img=img.convert('L') #convert to gray scale
            img=img.resize((8,8),Image.Resampling.LANCZOS)

            img_array= np.array(img) 

            img_array =255- img_array

            img_array= img_array/16.0  

            img_flattened= img_array.flatten() 

            final_features=[img_flattened]

            prediction= model.predict(final_features) 

            predicted_digit= int(prediction[0])

            return render_template('index.html', prediction_text= f"Predicted digit: {predicted_digit}")
        

        except Exception as e:
            return render_template('index.html', prediction_text="an error ")          
