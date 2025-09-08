from flask import *
from flask import Flask, render_template
import requests
import BildModels.vit as vit
from PIL import Image
import SprachModel.gpt2 as sp
import numpy as np

app = Flask(__name__, static_url_path='/frontend', static_folder='frontend')


def analyse(symptome):

    dig = sp.diag(sp.dies(symptome))

    return dig


def vitt(bild):
    image = Image.open(bild)

    dig = vit.process_ft_image(image)

    return dig


def resn(bild):

    dig = sp.diag(sp.dies(bild))

    return dig

# this mehtod will get the symptome from the page as json, and return a response
@app.route('/analyse', methods=['POST'])
def analyser():
    
    data = request.get_json().get('symptome')

    print(data)

    analysing = analyse(data)

    print(analysing)

    return jsonify(response=analysing)


@app.route('/transfor', methods=['POST'])
def transformer():
    
    uploaded_photo = request.files['photo']

    response = vitt(uploaded_photo)

    print(response)
    ed = response[0]
    if ed == 1:
        response = sp.diag(sp.dies("PNEUMONIA"))
    else:
        response = "No Signs of PNEUMONIA detected"

    return jsonify(response=response)


# Home Page
@app.route("/")
def home():
    return render_template('index.html')



if __name__ == '__main__':
    app.run("0.0.0.0", 9091, debug=True)
