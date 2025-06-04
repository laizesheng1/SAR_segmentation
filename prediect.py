from torchvision import transforms
from ai_test import  predict_and_show,get_image_mask_paths,UNet,evaluate,SegmentationDataset,DataLoader
import torch

if __name__ == '__main__':
    # 模型与设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    # 加载训练好的模型参数
    model.load_state_dict(torch.load('best_unet_5class.pth', map_location=device))
    print("已加载训练好的模型参数")

    # 示例路径（仅需要图像路径即可）
    val_image_dir = r"data\SAR\train"
    val_mask_dir = r"data\LAB\train"

    val_image_paths, val_mask_paths = get_image_mask_paths(val_image_dir, val_mask_dir)

    # val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, image_augment)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    # val_loss, val_dice = evaluate(model, val_loader, device)
    # print(f'Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f}')
    # 进行预测并可视化
    predict_and_show(model, val_image_paths[1], val_mask_paths[1], device)
