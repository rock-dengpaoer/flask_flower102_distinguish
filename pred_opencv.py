import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from PIL import Image 
import json
import cv2 as cv
import os
import sys
import numpy as np
import random
from pprint import pprint

def random_crop(image, crop_shape, padding=0):
    img_h = image.shape[0]
    img_w = image.shape[1]
    img_d = image.shape[2]

    oshape_h = img_h + 2 * padding
    oshape_w = img_w + 2 * padding
    img_pad = np.zeros([oshape_h, oshape_w, img_d], np.uint8)
    img_pad[padding:padding+img_h, padding:padding+img_w, 0:img_d] = image
  
    nh = random.randint(0, oshape_h - crop_shape[0])
    nw = random.randint(0, oshape_w - crop_shape[1])
    image_crop = img_pad[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]

    return image_crop

def main():

    with open("Oxford-102_Flower_dataset_labels.txt") as f:
        labels = f.readlines()
    f.close()
    label_list = []
    for label in labels:
        label_list.append(label.split("\n")[0].split("'")[1])

    checkpoint = torch.load('model_best.pth.tar')
    model = timm.create_model(checkpoint["arch"], num_classes = 102)
    model.load_state_dict(checkpoint["state_dict"])

    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        # transforms.Resize(size=235, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        # transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])

    result_list = []
    acc_list = []

    path = "./data/test/"
    
    for folders in os.listdir(path):
        img_paths = os.listdir(path + folders)
        acc = 0
        i = 0
        for img_path in img_paths:
            img = cv.imread(path + folders + "/" + img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            img = cv.resize(img, (235, 235), interpolation=cv.INTER_CUBIC)
            img = random_crop(img, (224, 224))
            # print(img.shape)
            # img = img.transpose(1, 0, 2)
            img_tensor = transform(img)
            img_tensor = img_tensor.unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor.to(device))
            maxk = min(max((1, 5)), output.size()[1])
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            print("label is {}".format(int(folders) - 1), end="\t")
            print("pred is ", end=" ")
            print(int(pred[0, 0]))

            result_list.append((int(folders) - 1, int(pred[0, 0])))
            if int(folders) - 1 == int(pred[0, 0]):
                acc = acc + 1
            i = i + 1
        acc_list.append((int(folders) - 1, acc / i))
    
    # print(acc / i)
    # print(result_list)
    # pprint(result_list)
    pprint(acc_list)

    sys.exit()


    img = Image.open("data/test/1/image_06734.jpg")
    img_tensor = transform(img)

    print(img_tensor)
    
    img_tensor = img_tensor.unsqueeze(0)

    
    with torch.no_grad():
        output = model(img_tensor.to(device))
    print(output)

    maxk = min(max((1, 5)), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    print(pred)

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

if __name__ == "__main__":
    main()