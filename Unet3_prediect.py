import numpy as np
import torch
from train import UNet3plus,get_data_list,UNet3Dataset,image_transform1,evaluate,show_result
from torch.utils.data import DataLoader

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3plus().cuda()
    model.eval()                 # 开启评估模式
    model.load_state_dict(torch.load('UNet_pdparams.pth', map_location=device)) # 载入预训练模型参数
    dice_accs = []
    # val_image_dir = r"data\SAR\test"
    # val_mask_dir = r"data\LAB\test"
    # val_image_dir = r"output\images\test"
    # val_mask_dir = r"output\labels\test"
    val_image_dir = r"FloodNet\val\image"
    val_mask_dir = r"FloodNet\val\label"
    val_data_list = get_data_list(val_image_dir, val_mask_dir)
    val_dataset = UNet3Dataset(val_data_list, image_transform1, augment=None)  # 测试集
    val_loader = DataLoader(val_dataset,  # 测试数据集
                             batch_size=1,  # 每批次的样本数
                             num_workers=4,  # 加载数据的子进程数
                             shuffle=False,  # 不打乱数据集
                             drop_last=False)  # 不丢弃不完整的样本批次

    # evaluate(model,val_loader)
    #预测结果可视化
    for batch_id, data in enumerate(val_loader):
        if batch_id == 0:
            image, label = data
            image,label= image.to(device), label.to(device)
            pred = model(image)
            image_path,label_path = val_data_list[batch_id]
            show_result(image_path, label_path, pred)
