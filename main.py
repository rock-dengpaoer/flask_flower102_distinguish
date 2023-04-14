import timm
import torch
import torchvision.models as models

def main():
    checkpoint = torch.load('model_best.pth.tar')
    print(type(checkpoint))
    for key, value in checkpoint.items():
        print(key)
        if(key == "epoch"):
            print(value)
        elif(key == "arch"):
            print(value)
    
    model = timm.create_model(checkpoint["arch"], num_classes = 102)
    model.load_state_dict(checkpoint["state_dict"])
    print(model)
    print("over")
        # print(value)
    # arch = checkpoint['arch']
    # model = models.__dict__[arch]()
    # # model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(checkpoint['state_dict'])
    # print(model)

if __name__ == "__main__":
    main()