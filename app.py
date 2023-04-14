from flask import Flask, flash, redirect, url_for
from markupsafe import escape
from flask import request
from werkzeug.utils import secure_filename
import timm
import timm.utils as utils
import timm.data as timm_data
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
import shutil
import os
import json

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg'}

checkpoint = torch.load('model_best.pth.tar')
model = timm.create_model(checkpoint["arch"], num_classes = 102)
model.load_state_dict(checkpoint["state_dict"])


device = torch.device('cuda:0')
model = model.to(device)
model.eval()

with open("Oxford-102_Flower_dataset_labels.txt") as f:
        labels = f.readlines()
f.close()
label_list = []
for label in labels:
    label_list.append(label.split("\n")[0].split("'")[1])



app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["CACHE_TYPE"] = "null"

  
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # 获取上传的文件
        file = request.files['file']
      
        if file and allowed_file(file.filename):
          
            # 获取文件属性
            filename = secure_filename(file.filename)
            _, file_type = filename.split('.')
            file.save(UPLOAD_FOLDER + filename)

            test_datasets = timm_data.create_dataset(name = "", root="./uploads")
            test_loader = timm_data.create_loader(test_datasets,
                                                   batch_size = 1, 
                                                   input_size=(3, 224, 224), 
                                                   interpolation="bicubic", 
                                                   mean=(0.485, 0.456, 0.406),
                                                   std=(0.229, 0.224, 0.225),
                                                   crop_pct=0.95,
                                                   )
            with torch.no_grad():
                for key, (input, label)  in enumerate(test_loader):
                    input = input.to(device)
                    output = model(input)
                    maxk = min(max((1, 5)), output.size()[1])
                    _, pred = output.topk(maxk, 1, True, True)
                    pred = pred.t()
                    print(pred)
                    print(pred.shape)
                    list = pred.to(torch.device('cpu')).numpy().tolist()
                    print(list)
                    result = []
                    key_list = []
                    for index in range(len(list)):
                        result.append(label_list[list[index][0]])
                        key_list.append(index + 1)
                    print(result)
                    reslut_dict = dict(zip(key_list, result))
                    print(reslut_dict)
                    reslut_dict_json = json.dumps(reslut_dict)
                    print(reslut_dict_json)
            shutil.rmtree("./uploads")
            os.mkdir("./uploads")
            return reslut_dict_json
        
    shutil.rmtree("./uploads")
    os.mkdir("./uploads")
    return "error"
