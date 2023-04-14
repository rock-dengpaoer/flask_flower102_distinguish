import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from PIL import Image 
import json

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
        transforms.Resize(size=235, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])

    img = Image.open("uploads/image_06734.jpg")
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