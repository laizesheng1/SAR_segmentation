import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
from torch.cuda.amp import autocast, GradScaler
from data_augmentation import image_augment

Image.MAX_IMAGE_PIXELS = None
IMAGE_SIZE=(1024, 1024)
# 定义颜色到类别的映射（示例）
COLOR_MAP = {
    (0, 0, 0): 0,  # 黑色→背景
    (255, 0, 0): 1,  # 红色→建筑物
    (0, 255, 0): 2,  # 绿色→植被
    (0, 0, 255): 3,  # 蓝色→水域
    (255, 255, 0): 4  # 黄色→道路
}
# ==================== 1. 数据加载 ====================
class SegmentationDataset(Dataset):
    '''
    image_paths: list of image paths
    mask_paths: list of mask paths
    transform: 数据增强
    '''

    def __init__(self, image_paths, mask_paths, transform, augment=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img, lab = self.transform(self.image_paths[idx], self.mask_paths[idx], self.augment)
        return img, lab
#把mask转为[0,4], image转为tensor
def image_transform(image_path, mask_path,augment=None):
    # 读取灰度图像
    image = Image.open(image_path).convert('L')  # PIL.image[H,W]
    rgb_mask = Image.open(mask_path).convert('RGB')
    image = image.resize(IMAGE_SIZE)
    rgb_mask = rgb_mask.resize(IMAGE_SIZE)
    # 数据增强   输入PIL.image
    if augment is not None:
        image, rgb_mask = augment(image, rgb_mask)

    rgb_mask = np.array(rgb_mask)  # ndarray[H,W,3]
    h, w, _ = rgb_mask.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)

    for color, class_id in COLOR_MAP.items():
        matches = np.all(rgb_mask == np.array(color).reshape(1, 1, 3), axis=2)
        class_mask[matches] = class_id

    image = np.array(image).astype("float32") / 255.0  # [H, W]
    image = torch.from_numpy(image).unsqueeze(0)
    # image = np.array(image).astype("float32").transpose((2, 0, 1))        #image[H W C]->[C H W]
    # image = torch.Tensor(image/255.0)
    class_mask = torch.from_numpy(class_mask).long()

    return image, class_mask

def get_image_mask_paths(image_dir, mask_dir, image_exts={".tif", ".png", ".jpg", ".jpeg"}):
    def get_sorted_files(directory):
        return sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[-1].lower() in image_exts
        ])

    image_paths = get_sorted_files(image_dir)
    mask_paths = get_sorted_files(mask_dir)

    if len(image_paths) != len(mask_paths):
        print(f"Warning: Number of images ({len(image_paths)}) and masks ({len(mask_paths)}) do not match!")

    return image_paths, mask_paths

# ==================== 2. UNet模型定义 ====================
#深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleDSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(卷积 => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),     #padding
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2,stride=2),         //什么时候使用stride=2
            nn.MaxPool2d(2),
            # DoubleConv(in_channels, out_channels)
            DoubleDSConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        # self.conv = DoubleConv(in_channels, out_channels)
        self.conv = DoubleDSConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=5, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 64)
        self.inc = DoubleDSConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        # self.up1 = Up(1024, 512, bilinear)
        # self.up2 = Up(512, 256, bilinear)
        # self.up3 = Up(256, 128, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        self.up1 = Up(1536, 512, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 128, bilinear)
        self.up4 = Up(192, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits  # 输出0-1之间的概率图


# ==================== 3. 训练配置 ====================
def multiclass_dice_coeff(pred, target, smooth=1):
    C = pred.shape[1]  # 类别数
    dice = 0.0
    for class_id in range(C):
        pred_flat = (pred.argmax(dim=1) == class_id).float().view(-1)
        target_flat = (target == class_id).float().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice += (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return dice / C  # 返回平均Dice

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.multi_class_dice_loss(inputs, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

    def multi_class_dice_loss(self, inputs, targets, smooth=1e-5):
        num_classes = inputs.shape[1]
        inputs = torch.softmax(inputs, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (inputs * targets_onehot).sum(dims)
        union = inputs.sum(dims) + targets_onehot.sum(dims)
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

# ==================== 5. 验证与指标计算 ====================
def dice_coeff(pred, target):
    smooth = 1.
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def evaluate(model, loader, device):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    class_weights = torch.tensor([0.1, 0.2, 0.2, 0.3, 0.4])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)       #[B channels H W]tensor
            masks = masks.to(device, non_blocking=True)         #[B H W]
            outputs = model(images)         #[B class_num H W]
            val_loss += criterion(outputs, masks).item()
            # dice_score += dice_coeff(outputs.round(), masks)
            dice_score += multiclass_dice_coeff(outputs,masks)
    # print("Eval \t Loss: %.5f \t Dice: %.5f" % (val_loss / len(loader), dice_score / len(loader)))
    return val_loss / len(loader), dice_score / len(loader)
# ==================== 6. 预测可视化 ====================
def predict_and_show(model, image_path, label_path, device):

    model.eval()
    image, label = image_transform(image_path, label_path)
    image, label = image.to(device), label.to(device)
    with torch.no_grad():
        logits = model(image[np.newaxis,...])
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()  # [H,W]

    # 将预测结果映射回RGB
    palette = np.array([
        [0, 0, 0],       # 背景→黑色
        [255, 0, 0],  # 类别1→红色
        [0, 255, 0],  # 类别2→绿色
        [0, 0, 255],  # 类别3→蓝色
        [255, 255, 0],  # 类别4→黄色
    ], dtype=np.uint8)

    pred_rgb = palette[pred]  # [H,W,3]
    image = Image.open(image_path).resize(IMAGE_SIZE)
    label = Image.open(label_path).resize(IMAGE_SIZE)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(label)
    plt.title('Ground Truth')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_rgb)
    plt.title('Predicted Mask')
    plt.show()


# ==================== 7. 主训练流程 ====================
if __name__ == '__main__':
    # 数据路径
    train_image_dir = r"data\SAR\train"
    train_mask_dir = r"data\LAB\train"
    val_image_dir = r"data\SAR\test"
    val_mask_dir = r"data\LAB\test"

    # 获取所有路径
    train_image_paths, train_mask_paths = get_image_mask_paths(train_image_dir, train_mask_dir)
    val_image_paths, val_mask_paths = get_image_mask_paths(val_image_dir, val_mask_dir)
    print(f"Found {len(train_image_paths)} training pairs")
    print(f"Found {len(val_image_paths)} validation pairs")

    # 获取数据
    train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, image_transform, image_augment)
    val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, image_transform, image_augment)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device('cuda')
    model = UNet().to(device)
    class_weights = torch.tensor([0.1, 0.2, 0.2, 0.2, 0.2])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # criterion = CombinedLoss(weight=class_weights.to(device), dice_weight=1.0, ce_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 150
    best_dice = 0.0
    accum_iter = 4
    optimizer.zero_grad()
    running_loss = 0.0
    scaler = GradScaler()
    # for epoch in range(num_epochs):
    #     train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    #     val_loss, val_dice = evaluate(model, val_loader, device)
    #
    #     print(f'Epoch {epoch + 1}/{num_epochs}')
    #     print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f}')
    #
    #     # 保存最佳模型
    #     if val_dice > best_dice:
    #         best_dice = val_dice
    #         torch.save(model.state_dict(), 'best_unet_5class.pth')
    #         print('Model saved!')
    for epoch in range(num_epochs):
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)  # [B, C, H, W]
                loss = criterion(outputs, masks)  # dice 或 crossentropy
            # loss.backward()
            loss = loss/accum_iter
            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (i + 1) % accum_iter == 0 or (i + 1) == len(train_loader):
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        # val_loss, val_dice = evaluate(model, train_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {running_loss / len(train_loader):.4f}')
        running_loss = 0.0  # 重置loss累积
    torch.save(model.state_dict(), 'best_unet_5class.pth')
    print('Model saved!')
    val_loss, val_dice = evaluate(model, train_loader, device)      #测试集数据一张图片太大
    print(f'Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f}')