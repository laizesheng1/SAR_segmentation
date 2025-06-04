import random
import numpy as np
import torch,cv2,os
from torch import sigmoid, nn, Tensor
from torch.nn import functional as F
from typing import Union, Tuple
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from init_weight import init_weights
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data_augmentation import image_augment

Image.MAX_IMAGE_PIXELS = None
BATCH_SIZE = 4           # 每批次的样本数
EPOCHS = 150              # 模型训练的总轮数
LOG_GAP = 500            # 输出训练信息的间隔
N_CLASSES = 5            # 图像分类种类数量
IMG_SIZE = (512, 512)    # 图像缩放尺寸
INIT_LR = 3e-4           # 初始学习率
MODEL_PATH = "UNet_pdparams.pth"  # 模型参数保存路径

class UNet3Dataset(Dataset):
    '''
    data_list: 图像路径、标签路径
    transform: 图像变换、image_tranform
    augment: 数据增强 image_augment
    '''
    def __init__(self, data_list ,transform ,augment) -> None:
        super(Dataset, self).__init__()
        random.shuffle(data_list)       #打乱训练的顺序
        self.data_list = data_list
        self.transform = transform
        self.augment = augment
    def __getitem__(self, idx):
        img_path, lab_path = self.data_list[idx]
        img,lab=self.transform(img_path, lab_path, self.augment)
        return img,lab
    def __len__(self):
        return len(self.data_list)

COLOR_MAP = {
    (0, 0, 0): 0,  # 背景
    (255, 0, 0): 1,       # 建筑物
    (0, 255, 0): 2,       # 植被
    (0, 0, 255): 3,       # 水域
    (255, 255, 0): 4,      # 道路
}

def convert_mask(mask_img):
    """将RGB彩色标签图转换为单通道整数 mask"""
    mask_np = np.array(mask_img)  # shape [H, W, 3]
    label_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.int64)

    for color, class_idx in COLOR_MAP.items():
        match = np.all(mask_np == color, axis=-1)
        label_mask[match] = class_idx

    return label_mask       #[H,W]
def image_transform(img_path, lab_path, augmentation=None):
    #1. 读取图像
    img = Image.open(img_path).convert('L')  # SAR灰度图
    lab_img = Image.open(lab_path).convert('RGB')  # 标签图为 RGB

    # Resize
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    lab_img = lab_img.resize(IMG_SIZE, Image.NEAREST)

    #2. 数据增强
    if augmentation is not None:
        img, lab_img = augmentation(img, lab_img)

    #3. 图像处理：转为 Tensor, 归一化
    img = np.array(img).astype("float32") / 255.0  # [H, W]
    img = torch.from_numpy(img).unsqueeze(0)  # 添加通道 -> [1, H, W]

    #4. 标签图：颜色 → 类别索引
    label_mask = convert_mask(lab_img)      #[H,W]
    lab = torch.from_numpy(label_mask).long()  # shape [H, W]

    return img, lab


def get_data_list(image_dir, mask_dir, image_exts={".tif", ".png", ".jpg", ".jpeg"}):
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

    data_list = list(zip(image_paths, mask_paths))
    return data_list

class Decoder(nn.Module):
    ''' Decoder for UNet3+ '''

    def __init__(self,
                 cur_stage: int,
                 cat_size: int,
                 up_size: int,
                 filters: list,
                 ks=3, s=1, p=1):
        ''' Args:
            * `cur_stage`: 当前解码器所在层数
            * `cat_size`: 统一后的特征图通道数
            * `up_size`: 特征融合后的通道总数
            * `filters`: 各卷积网络的卷积核数
            * `ks`: 卷积核大小 (默认为3)
            * `s`: 卷积运算步长 (默认为1)
            * `p`: 卷积填充大小 (默认为1)
        '''
        super(Decoder, self).__init__()
        self.n = len(filters)      # 卷积网络模块的个数

        for idx, num in enumerate(filters):
            idx += 1               # 待处理输出所在层数
            if idx < cur_stage:
                # he[idx]_PT_hd[cur_stage], Pool [ps] times
                ps = 2 ** (cur_stage - idx)
                block = nn.Sequential(nn.MaxPool2d(ps, ps, ceil_mode=True),
                                      nn.Conv2d(num, cat_size, ks, s, p),
                                      nn.BatchNorm2d(cat_size),
                                      nn.ReLU(inplace=True))
            elif idx == cur_stage:
                # he[idx]_Cat_hd[cur_stage], Concatenate
                block = nn.Sequential(nn.Conv2d(num, cat_size, ks, s, p),
                                      nn.BatchNorm2d(cat_size),
                                      nn.ReLU(inplace=True))
            else:
                # hd[idx]_UT_hd[cur_stage], Upsample [us] times
                us = 2 ** (idx - cur_stage)
                num = num if idx == 5 else up_size
                block = nn.Sequential(nn.Upsample(scale_factor=us, mode="bilinear"),
                                      nn.Conv2d(num, cat_size, ks, s, p),
                                      nn.BatchNorm2d(cat_size),
                                      nn.ReLU(inplace=True))
            setattr(self, "block%d" % idx, block)

        # fusion(he[]_PT_hd[], ..., he[]_Cat_hd[], ..., hd[]_UT_hd[])
        self.fusion = nn.Sequential(nn.Conv2d(up_size, up_size, ks, s, p),
                                    nn.BatchNorm2d(up_size),
                                    nn.ReLU(inplace=True))

        for m in self.children():       # 初始化各层网络的系数
            init_weights(m, init_type="kaiming")

    def forward(self, inputs: Tensor):
        outputs = []       # 记录各层的输出，以便于拼接起来
        for i in range(self.n):
            block = getattr(self, "block%d" % (i+1))
            outputs.append(block(inputs[i]))
        hd = self.fusion(torch.cat(outputs, 1))
        return hd
class Encoder(nn.Module):
    ''' Encoder for UNet3+ '''

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 is_batchnorm: bool,
                 n=2, ks=3, s=1, p=1):
        ''' Args:
            * `in_size`: 输入通道数
            * `out_size`: 输出通道数
            * `is_batchnorm`: 是否批正则化
            * `n`: 卷积层数量 (默认为2)
            * `ks`: 卷积核大小 (默认为3)
            * `s`: 卷积运算步长 (默认为1)
            * `p`: 卷积填充大小 (默认为1)
        '''
        super(Encoder, self).__init__()
        self.n = n

        for i in range(1, self.n+1):    # 定义多层卷积神经网络
            if is_batchnorm:
                block = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                      nn.BatchNorm2d(out_size),
                                      nn.ReLU(inplace=True))
            else:
                block = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                      nn.ReLU(inplace=True))
            setattr(self, "block%d" % i, block)
            in_size = out_size

        for m in self.children():   # 初始化各层网络的系数
            init_weights(m, init_type="kaiming")

    def forward(self, x: Tensor):
        for i in range(1, self.n+1):
            block = getattr(self, "block%d" % i)
            x = block(x)            # 进行前向传播运算
        return x

class UNet3plus(nn.Module):
    '''
    """reference"""
    * Authors: Huimin Huang, et al (2020)
    * Paper: A Full-Scale Connected UNet for Medical Image Segmentation
    * Link: https://arxiv.org/pdf/2004.08790.pdf
    '''

    def __init__(self,
                 in_channels: int = 1,
                 n_classes: int = 5,
                 is_batchnorm: bool = True,
                 deep_sup: bool = False,
                 set_cgm: bool = False):
        ''' Args:
            * `in_channels`: Number of input channels.
            * `n_classes`: Number of classes.
            * `is_batchnorm`: Whether using batch normalization.
            * `deep_sup`: Whether using Deep Supervision.
            * `set_cgm`: Whether using Class-guided Module.
        '''
        super(UNet3plus, self).__init__()
        self.deep_sup = deep_sup
        self.set_cgm = set_cgm
        filters = [64, 128, 256, 512, 1024]      # 各模块的卷积核大小
        cat_channels = filters[0]                # 统一后的特征图通道数
        cat_blocks = 5                           # 编（解）码器的层数
        up_channels = cat_channels * cat_blocks  # 特征融合后的通道数

        # ====================== Encoders ======================
        self.conv_e1 = Encoder(in_channels, filters[0], is_batchnorm)
        self.pool_e1 = nn.MaxPool2d(kernel_size=2)
        self.conv_e2 = Encoder(filters[0], filters[1], is_batchnorm)
        self.pool_e2 = nn.MaxPool2d(kernel_size=2)
        self.conv_e3 = Encoder(filters[1], filters[2], is_batchnorm)
        self.pool_e3 = nn.MaxPool2d(kernel_size=2)
        self.conv_e4 = Encoder(filters[2], filters[3], is_batchnorm)
        self.pool_e4 = nn.MaxPool2d(kernel_size=2)
        self.conv_e5 = Encoder(filters[3], filters[4], is_batchnorm)

        # ====================== Decoders ======================
        self.conv_d4 = Decoder(4, cat_channels, up_channels, filters)
        self.conv_d3 = Decoder(3, cat_channels, up_channels, filters)
        self.conv_d2 = Decoder(2, cat_channels, up_channels, filters)
        self.conv_d1 = Decoder(1, cat_channels, up_channels, filters)

        # ======================= Output =======================
        if self.set_cgm:
            # -------------- Class-guided Module ---------------
            self.cls = nn.Sequential(nn.Dropout(p=0.5),
                                     nn.Conv2d(filters[4], 2, 1),
                                     nn.AdaptiveMaxPool2d(1),
                                     nn.Sigmoid())
        if self.deep_sup:
            # -------------- Bilinear Upsampling ---------------
            self.upscore5 = nn.Upsample(scale_factor=16, mode="bilinear")
            self.upscore4 = nn.Upsample(scale_factor=8, mode="bilinear")
            self.upscore3 = nn.Upsample(scale_factor=4, mode="bilinear")
            self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")
            # ---------------- Deep Supervision ----------------
            self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, 1, 1)
            self.outconv4 = nn.Conv2d(up_channels, n_classes, 3, 1, 1)
            self.outconv3 = nn.Conv2d(up_channels, n_classes, 3, 1, 1)
            self.outconv2 = nn.Conv2d(up_channels, n_classes, 3, 1, 1)
        self.outconv1 = nn.Conv2d(up_channels, n_classes, 3, 1, 1)

        # ================= Initialize Weights =================
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dot_product(self, seg: Tensor, cls: Tensor):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        clssp = torch.ones((1, N))
        ecls = (cls * clssp).view(B, N, 1)
        final = (seg * ecls).view(B, N, H, W)
        return final

    def forward(self, x: Tensor) -> tuple:
        # ====================== Encoders ======================
        e1 = self.conv_e1(x)                  # e1: 320*320*64
        e2 = self.pool_e1(self.conv_e2(e1))   # e2: 160*160*128
        e3 = self.pool_e2(self.conv_e3(e2))   # e3: 80*80*256
        e4 = self.pool_e3(self.conv_e4(e3))   # e4: 40*40*512
        e5 = self.pool_e4(self.conv_e5(e4))   # e5: 20*20*1024

        # ================ Class-guided Module =================
        if self.set_cgm:
            cls_branch = self.cls(e5).squeeze(3).squeeze(2)
            cls_branch_max = cls_branch.argmax(dim=1)
            cls_branch_max = cls_branch_max[:, np.newaxis].float()

        # ====================== Decoders ======================
        d5 = e5
        d4 = self.conv_d4((e1, e2, e3, e4, d5))
        d3 = self.conv_d3((e1, e2, e3, d4, d5))
        d2 = self.conv_d2((e1, e2, d3, d4, d5))
        d1 = self.conv_d1((e1, d2, d3, d4, d5))

        # ======================= Output =======================
        if self.deep_sup:
            y5 = self.upscore5(self.outconv5(d5))   # 16 => 256
            y4 = self.upscore4(self.outconv4(d4))   # 32 => 256
            y3 = self.upscore3(self.outconv3(d3))   # 64 => 256
            y2 = self.upscore2(self.outconv2(d2))   # 128 => 256
            y1 = self.outconv1(d1)                  # 256
            if self.set_cgm:
                y5 = self.dot_product(y5, cls_branch_max)
                y4 = self.dot_product(y4, cls_branch_max)
                y3 = self.dot_product(y3, cls_branch_max)
                y2 = self.dot_product(y2, cls_branch_max)
                y1 = self.dot_product(y1, cls_branch_max)
            return (sigmoid(y1), sigmoid(y2), sigmoid(y3),
                    sigmoid(y4), sigmoid(y5))
        else:
            y1 = self.outconv1(d1)                  # 320*320*n_classes
            if self.set_cgm:
                y1 = self.dot_product(y1, cls_branch_max)
            return sigmoid(y1)


class DiceLoss(nn.Module):
    ''' Dice Loss for Segmentation Tasks'''

    def __init__(self,
                 n_classes: int = 5,
                 smooth: Union[float, Tuple[float, float]] = (0, 1e-6),
                 sigmoid_x: bool = False,
                 softmax_x: bool = True,
                 onehot_y: bool = True,
                 square_xy: bool = True,
                 include_bg: bool = False,
                 reduction: str = "mean"):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `sigmoid_x`: whether using `sigmoid` to process the result.
        * `softmax_x`: whether using `softmax` to process the result.
        * `onehot_y`: whether using `one-hot` to encode the label.
        * `square_xy`: whether using squared result and label.
        * `include_bg`: whether taking account of bg-class when computering dice.
        * `reduction`: reduction function of dice loss.
        '''
        super(DiceLoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise NotImplementedError(
                "`reduction` of dice loss should be 'mean' or 'sum'!"
            )
        if isinstance(smooth, float):
            self.smooth = (smooth, smooth)
        else:
            self.smooth = smooth

        self.n_classes = n_classes
        self.sigmoid_x = sigmoid_x
        self.softmax_x = softmax_x
        self.onehot_y = onehot_y
        self.square_xy = square_xy
        self.include_bg = include_bg
        self.reduction = reduction

    def forward(self, pred, mask):
        #add
        if isinstance(pred, tuple):
            pred = pred[0]
        (sm_nr, sm_dr) = self.smooth

        if self.sigmoid_x:
            pred = F.sigmoid(pred)
        if self.n_classes > 1:
            if self.softmax_x and self.n_classes == pred.shape[1]:
                pred = F.softmax(pred, dim=1)
            if self.onehot_y:
                mask = mask if mask.ndim < 4 else mask.squeeze(axis=1)
                mask = F.one_hot(mask.long(), self.n_classes)
                mask = mask.permute((0, 3, 1, 2))
            if not self.include_bg:
                pred = pred[:, 1:] if pred.shape[1] > 1 else pred
                mask = mask[:, 1:] if mask.shape[1] > 1 else mask
        if pred.ndim != mask.ndim or pred.shape[1] != mask.shape[1]:
            raise ValueError(
                f"The shape of `pred`({pred.shape}) and " +
                f"`mask`({mask.shape}) should be the same."
            )

        # only reducing spatial dimensions:
        reduce_dims = torch.arange(2, pred.ndim).tolist()
        insersect = torch.sum(pred * mask, dim=reduce_dims)
        if self.square_xy:
            pred, mask = torch.pow(pred, 2), torch.pow(mask, 2)
        pred_sum = torch.sum(pred, dim=reduce_dims)
        mask_sum = torch.sum(mask, dim=reduce_dims)
        loss = 1. - (2 * insersect + sm_nr) / (pred_sum + mask_sum + sm_dr)

        if self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

def dice_func(pred: np.ndarray, mask: np.ndarray,
         n_classes: int, ignore_bg: bool = False, smooth: float = 1e-6):
    ''' compute dice (for NumpyArray) '''
    # def sub_dice(x: torch.Tensor, y: torch.Tensor, sm: float = 1e-6):
    #     intersect = np.sum(np.sum(np.sum(x * y)))
    #     y_sum = np.sum(np.sum(np.sum(y)))
    #     x_sum = np.sum(np.sum(np.sum(x)))
    #     return (2 * intersect + sm) / (x_sum + y_sum + sm)
    #
    # assert pred.shape == mask.shape
    # assert isinstance(ignore_bg, bool)
    # return [
    #     sub_dice(pred==i, mask==i)
    #     for i in range(int(ignore_bg), n_classes)
    # ]
    ''' Compute Dice scores per class for single image
        Args:
            pred: (H, W), predicted class indices (after argmax)
            mask: (H, W), ground truth class indices
            n_classes: total number of classes
            ignore_bg: whether to ignore background class (class 0)
            smooth: smoothing factor to avoid divide-by-zero
        Returns:
            list of dice score per class
    '''
    assert pred.shape == mask.shape, "Shape mismatch between pred and mask"
    scores = []
    for i in range(int(ignore_bg), n_classes):
        pred_i = (pred == i).astype(np.float32)
        mask_i = (mask == i).astype(np.float32)

        intersection = np.sum(pred_i * mask_i)
        union = np.sum(pred_i) + np.sum(mask_i)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        scores.append(dice)

    return scores

def evaluate(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # 开启评估模式
    model.load_state_dict(torch.load('UNet_pdparams.pth', map_location=device))  # 载入预训练模型参数
    model.to(device)
    dice_accs = []

    for batch_id, data in enumerate(test_loader):
        image, label = data
        image,label= image.to(device), label.to(device)
        pred = model(image)  # 预测结果
        pred = pred.argmax(axis=1).squeeze(axis=0).cpu().numpy()        #[H,W]
        label = label.squeeze(0).squeeze(0).cpu().numpy()           #[H, W]
        dice = dice_func(pred, label, N_CLASSES)  # 计算损失函数值
        dice_accs.append(np.mean(dice))
    print("Eval \t Dice: %.5f" % (np.mean(dice_accs)))

INDEX2COLOR = {v: k for k, v in COLOR_MAP.items()}
def show_result(img_path, lab_path, pred):
    ''' 展示原图、标签以及预测结果 '''
    #pred: [batch ,Class, H ,W]
    def add_subimg(img, loc, title):
        plt.subplot(loc)
        plt.title(title)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    def decode_segmap(mask):
        """将预测的类别 mask 映射为 RGB 彩色图"""
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in INDEX2COLOR.items():
            color_mask[mask == class_idx] = color
        return color_mask

    # 加载原图和标签图
    img = Image.open(img_path).resize(IMG_SIZE)
    lab = Image.open(lab_path).resize(IMG_SIZE)
    lab = np.array(lab)

    # 处理预测结果：取每个像素最大概率的类别索引
    pred_class = pred.squeeze(0).argmax(dim=0)  # [256, 256]
    pred_class_np = pred_class.cpu().numpy()  # 转 numpy

    # 映射为 RGB 彩图
    pred_color = decode_segmap(pred_class_np)

    # 显示图像
    plt.figure(figsize=(12, 4))
    add_subimg(img, 131, "Image")
    add_subimg(lab, 132, "Label")
    add_subimg(pred_color, 133, "Predict")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # 训练初始化
    train_image_dir = r"data\SAR\train"
    train_mask_dir = r"data\LAB\train"
    val_image_dir = r"data\SAR\test"
    val_mask_dir = r"data\LAB\test"
    train_data_list = get_data_list(train_image_dir, train_mask_dir)
    test_data_list = get_data_list(val_image_dir, val_mask_dir)
    print("train_data_list is :", train_data_list.__len__())
    print("test_data_list is :", test_data_list.__len__())
    train_dataset = UNet3Dataset(train_data_list, image_transform, image_augment)  # 训练集
    test_dataset = UNet3Dataset(test_data_list, image_transform, augment=None)  # 测试集
    train_loader = DataLoader(train_dataset,  # 训练数据集
                              batch_size=BATCH_SIZE,  # 每批次的样本数
                              num_workers=4,  # 加载数据的子进程数
                              shuffle=True,  # 打乱数据集
                              drop_last=False)  # 不丢弃不完整的样本批次

    test_loader = DataLoader(test_dataset,  # 测试数据集
                             batch_size=1,  # 每批次的样本数
                             num_workers=4,  # 加载数据的子进程数
                             shuffle=False,  # 不打乱数据集
                             drop_last=False)  # 不丢弃不完整的样本批次
    # print("training pairs:", train_loader.__len__())
    # print("validation pairs:", test_loader.__len__())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3plus(n_classes=N_CLASSES, deep_sup=False, set_cgm=False).cuda()

    model.train()  # 开启训练模式
    # 定义Adam优化器
    optimizer =  optim.Adam(
        model.parameters(),
        lr=INIT_LR,
        weight_decay=1e-5
    )
    # 定义学习率衰减器
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=EPOCHS,
    )
    dice_loss = DiceLoss(n_classes=N_CLASSES)
    loss_list = []  # 用于可视化

    for ep in range(EPOCHS):
        ep_loss_list = []
        for batch_id, data in enumerate(train_loader):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            pred = model(image)  # 预测结果
            # print(type(image), type(label), type(pred[0]))
            # print(image.shape, label.shape, pred[0].shape)
            loss = dice_loss(pred, label)  # 计算损失函数值
            if batch_id % LOG_GAP == 0:  # 定期输出训练结果
                print("Epoch：%2d，Batch：%3d，Loss：%.5f" % (ep, batch_id, loss))
            ep_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 衰减一次学习率
        loss_list.append(np.mean(ep_loss_list))
        print("【Train】Epoch：%2d，Loss：%.5f" % (ep, loss_list[-1]))
    torch.save(model.state_dict(), MODEL_PATH)  # 保存训练好的模型

    evaluate(model, test_loader)
    #训练过程可视化
    fig = plt.figure(figsize=[10, 5])

    # 训练误差图像：
    ax = fig.add_subplot(111, facecolor="#E8E8F8")
    ax.set_xlabel("Steps", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)
    plt.tick_params(labelsize=14)
    ax.plot(range(len(loss_list)), loss_list, color="orangered")
    ax.grid(linewidth=1.5, color="white")  # 显示网格

    fig.tight_layout()
    plt.show()
    plt.close()
