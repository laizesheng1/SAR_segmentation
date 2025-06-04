import numpy as np
import torch
from train import UNet3plus,get_data_list,UNet3Dataset,image_transform,evaluate,show_result
from torch.utils.data import DataLoader

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3plus().cuda()
    model.eval()                 # 开启评估模式
    model.load_state_dict(torch.load('UNet_pdparams.pth', map_location=device)) # 载入预训练模型参数
    dice_accs = []
    val_image_dir = r"data\SAR\train"
    val_mask_dir = r"data\LAB\train"
    test_data_list = get_data_list(val_image_dir, val_mask_dir)
    test_dataset = UNet3Dataset(test_data_list, image_transform, augment=None)  # 测试集
    test_loader = DataLoader(test_dataset,  # 测试数据集
                             batch_size=1,  # 每批次的样本数
                             num_workers=4,  # 加载数据的子进程数
                             shuffle=False,  # 不打乱数据集
                             drop_last=False)  # 不丢弃不完整的样本批次

    # evaluate(model,test_loader)
    #预测结果可视化
    for batch_id, data in enumerate(test_loader):
        if batch_id == 0:
            image, label = data
            image,label= image.to(device), label.to(device)
            pred = model(image)
            image_path,label_path = test_data_list[batch_id]
            show_result(image_path, label_path, pred)
