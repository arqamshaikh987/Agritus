# Importing essential libraries and modules
import os
from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from markupsafe import Markup
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# RandomForest model is already pre-trained and saved in 'models/RandomForest.pkl'
# Training it on every startup is unnecessary and resource-intensive, so it is commented out.
# df = pd.read_csv('Data/crop_recommendation.csv')
# X = df.drop(columns=['label'])
# y = df['label']
# model = RandomForestClassifier()
# model.fit(X, y)
# with open('models/RandomForest.pkl', 'wb') as file:
#     pickle.dump(model, file)

# -----------------------------------------------------------------------
# ResNet9 model architecture (must match the training notebook exactly)
# -----------------------------------------------------------------------

class ImageClassificationBase(nn.Module):
    pass

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)      # 128 x 64 x 64
        self.res1  = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)     # 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True)     # 512 x 4 x 4
        self.res2  = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

# Loading plant disease classification model (PyTorch .pth)
disease_model_path = 'models/plant_disease_model.pth'
_device = torch.device('cpu')
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=_device))
disease_model.eval()

# Loading crop recommendation model
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img):
    """
    Transforms image and predicts disease label using PyTorch ResNet9
    :params: image bytes
    :return: prediction (string)
    """
    # Open image and ensure RGB mode
    image = Image.open(io.BytesIO(img)).convert('RGB')
    # Resize to exactly 256x256 (expected by the model architecture)
    image = image.resize((256, 256))
    # Convert to numpy array, scale to [0, 1]
    img_arr = np.array(image, dtype=np.float32) / 255.0
    # Reorder dimensions from HWC to CHW, add batch dimension: BCHW
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = disease_model(img_tensor)

    # Extract prediction index and map to class label
    pred_idx = torch.argmax(outputs, dim=1).item()
    prediction = disease_classes[pred_idx]
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
app.secret_key = 'secret_key'

# render home page

@app.route('/')
def home():
    title = 'Agritus - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page

@app.route('/crop-recommend')
def crop_recommend():
    title = 'Agritus - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page

@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Agritus - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)


# ===============================================================================================
# RENDER PREDICTION PAGES
# render crop recommendation result page


@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Agritus - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:
            return render_template('try_again.html', title=title)

# render fertilizer recommendation result page

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Agritus - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Agritus - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

if __name__ == '__main__':
    app.run()
