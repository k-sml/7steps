from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms
from pytorch_pretrained_vit import ViT
import json
import os

app = Flask(__name__)
app.config['MODEL'] = ViT('B_16_imagenet1k', pretrained=True)
app.config['UPLOAD_PATH'] = './static'

@app.route('/')    # http://xxx 以降のURLパスを '/' と指定
def index():
    fuga = "123"
    return render_template('index.html', hoge=fuga)   #defalutではtemplatesの直下のindex.htmlを見に行くことになっている

@app.route('/upload', methods = ['post'])
def upload():
    img_file = request.files['img_file']
    filename = secure_filename(img_file.filename)
    ul_path = f"{app.config['UPLOAD_PATH']}/{filename}"
    if os.path.exists(app.config['UPLOAD_PATH']) != True:
        os.mkdir(app.config['UPLOAD_PATH'])
    img_file.save(ul_path)
    return render_template("index.html", img_url=ul_path)

@app.route('/recognition', methods=['post'])
def recognition():
    img_path = request.form['img_path']
    img = Image.open(img_path)
    tfms = transforms.Compose([
        transforms.Resize(app.config['MODEL'].image_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    img = tfms(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        outputs = app.config['MODEL'](img)
    pred = torch.argmax(outputs)
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[key] for key in labels_map]
    pred_label = labels_map[pred]
    return render_template('index.html', img_url=img_path, pred_label=pred_label)

if __name__ == "__main__":
    app.run(debug = True)