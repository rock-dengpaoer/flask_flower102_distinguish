import torch.onnx
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import netron

def main():
    checkpoint = torch.load('model_best.pth.tar')
    model = timm.create_model(checkpoint["arch"], num_classes = 102)
    model.load_state_dict(checkpoint["state_dict"])

    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    print(model)



    torch.onnx.export(model,  # 待转换的网络模型和参数
                torch.randn(1, 3, 224, 224, device='cuda'), # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                "flower102_resnet50.onnx",  # 输出文件的名称
                verbose=False,      # 是否以字符串的形式显示计算图
                input_names=["input"],# + ["params_%d"%i for i in range(120)],  # 输入节点的名称，这里也可以给一个list，list中名称分别对应每一层可学习的参数，便于后续查询
                output_names=["output"], # 输出节点的名称
                opset_version=10,   # onnx 支持采用的operator set, 应该和pytorch版本相关，目前我这里最高支持10
                do_constant_folding=True, # 是否压缩常量
                )
    
    netron.start("flower102_resnet50.onnx")
    






if __name__ == "__main__":
    main()