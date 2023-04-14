import timm
import timm.utils as utils
import timm.data as timm_data
import torch
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

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


    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()

    transform  = transforms.Compose([
        transforms.RandomResizedCrop(224),
        #  transforms.RandomHorizontalFlip(),
        # transforms.Resize(224),
        transforms.ToTensor(),
    ])

    test_datasets = timm_data.create_dataset(name = "", root="data/test")
    # transform = timm_data.create_transform(input_size=(3, 224, 224), 
    #                            interpolation="bicubic", 
    #                            mean=(0.485, 0.456, 0.406),
    #                            std=(0.229, 0.224, 0.225),
    #                            crop_pct=0.95,
    #                            crop_mode="center")
    print(test_datasets)
    test_loader = timm_data.create_loader(test_datasets,
                                          batch_size = 1, 
                                input_size=(3, 224, 224), 
                                interpolation="bicubic", 
                               mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225),
                               crop_pct=0.95,
                               )

    # test_loader = DataLoader(test_datasets, batch_size=16, shuffle = False, num_workers=4)
    
    with torch.no_grad():
        acc = 0
        acc_5 = 0
        for key, (input, label)  in enumerate(test_loader):
            # print(input)
            # print(label)
            input = input.to(device)
            # print(input)
            output = model(input)
            # pred = torch.max(output, dim=1)[1]
            # acc += torch.eq(pred, label.to(device)).sum().item()
            acc1, acc5 = utils.accuracy(output, label.to(device), topk=(1, 5))
            acc = acc + acc1
            acc_5 = acc_5 + acc5

            maxk = min(max((1, 5)), output.size()[1])
            batch_size = label.to(device).size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(label.to(device).reshape(1, -1).expand_as(pred))
            # return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in (1, )]
            print("label is {},".format(label))
            print(" pred is {}".format(pred))
        
            print("acc1 is {}, acc5 is {}".format(acc1, acc5))
            print(key)
            
            # break
        print(acc / (key + 1))
        print(acc_5 / (key + 1))
        
    


    print("over")
        # print(value)
    # arch = checkpoint['arch']
    # model = models.__dict__[arch]()
    # # model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(checkpoint['state_dict'])
    # print(model)

if __name__ == "__main__":
    main()