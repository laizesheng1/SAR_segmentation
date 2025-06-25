from data_augmentation import image_augment
from ai_test import  image_transform1,get_image_mask_paths,UNet,evaluate,SegmentationDataset,DataLoader,predict_and_show
import torch

if __name__ == '__main__':
    # 模型与设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    # 加载训练好的模型参数
    model.load_state_dict(torch.load('best_unet_10class_512_50_16_True.pth', map_location=device))
    print("已加载训练好的模型参数")

    # 示例路径（仅需要图像路径即可）
    # val_image_dir = r"data\SAR\test"
    # val_mask_dir = r"data\LAB\test"
    # val_image_dir = r"output\images\test"
    # val_mask_dir = r"output\labels\test"
    val_image_dir = r"FloodNet\val\image"
    val_mask_dir = r"FloodNet\val\label"

    val_image_paths, val_mask_paths = get_image_mask_paths(val_image_dir, val_mask_dir)

    # val_dataset = SegmentationDataset(val_image_paths, val_mask_paths,image_transform1, image_augment)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
    # val_loss, val_dice ,val_iou= evaluate(model, val_loader, device)
    # print(f'Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}')
    # 进行预测并可视化
    predict_and_show(model, val_image_paths[5], val_mask_paths[5], device)
