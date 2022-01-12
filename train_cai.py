# **************************************************# ************************************************************
import os
# 设备设定
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import (
    ZNormalization,
)
from tqdm import tqdm
from torchvision import utils

# from utils.metric import metric
import nibabel as nib
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# **************************************************# ************************************************************
# 文件目录传输
source_train_0_dir = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\3dclassification\source_train_1_dir'
source_train_1_dir = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\3dclassification\source_train_0_dir'
source_test_0_dir = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\3dclassification\source_test_0_dir'
source_test_1_dir = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\3dclassification\source_test_1_dir'
val_0 = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\3dclassification\\val0'
val_1 = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\3dclassification\\val1'
output_dir = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\3dclassification\caihaihua log'
ckpt = False
batch_size = 1
# **************************************************# ************************************************************
def train():


# ************************************************************# ************************************************************
    os.makedirs(output_dir, exist_ok=True)#创建多层目录(output目录)

    from densenet3dcai import generate_model
    model = generate_model(121, n_input_channels=1, num_classes=2, drop_rate=0.7)

# ************************************************************# ************************************************************
    model = torch.nn.DataParallel(model, device_ids=devicess)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0)#L2正则
    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
    scheduler = CosineAnnealingLR(optimizer, T_max=32)
# ************************************************************# ************************************************************
# *****************************************载于预权重*************************************************************************
    ckpt = None#预权重载入开关
    if ckpt is not None:
        print("load model:", ckpt)
        print(os.path.join(output_dir, 'checkpoint_latest.pt'))
        ckpt = torch.load(os.path.join(output_dir, 'checkpoint_latest.pt'),
                          map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"]) #学习率调节机制载入开关
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()
# ************************************************************# ************************************************************
#损失函数选择
#     from loss_function import Classification_Loss
    from loss_function import FocalLoss1
#     criterion = nn.CrossEntropyLoss()
#     criterion = Classification_Loss().cuda()
    criterion = FocalLoss1().cuda()
# ************************************************************# ************************************************************
    writer = SummaryWriter(r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\3dclassification\caihaihua log')
# ************************************************************# ************************************************************
    from data_function import MedData_train
    train_dataset = MedData_train(source_train_0_dir, source_train_1_dir)
    train_loader = DataLoader(train_dataset.training_set,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    model.train()


    from data_function import MedData_val

    val_dataset = MedData_val(val_0, val_1)
    val_loader = DataLoader(val_dataset.testing_set,
                            batch_size=1,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    model.eval()
# ************************************************************# ************************************************************
    epochs = 500 - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)
    j=0
    x_axis=[]

    t_loss = []
    v_loss = []



# ************************************************************# ************************************************************

    for epoch in range(1, epochs + 1):
        print("epoch:" + str(epoch))
        epoch += elapsed_epochs
        totalloss = 0
        val_totalloss = 0
        j += 1

        num_iters = 0

        gts = []
        predicts = []
        predicts1 = []
        gts1 = []

        for i, batch in enumerate(train_loader):
            model.train()


            # print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")
            optimizer.zero_grad()

            x = batch['source']['data']
            y = batch['label']


            # x = x.type(torch.FloatTensor).cuda()
            # y = y.type(torch.LongTensor).cuda()
            x = np.array(x)
            x = x[0][0]
            # print(x.type)
            print(x.shape)
            # if j = epoch:
            #     x = x.squeeze(-1)
            #     x = x[:, :1, :, :]
            #     print(x[0][0][0][0])
            array_img = nib.Nifti1Image(x,None)
            nib.save(array_img, f'{i}.nii.gz')
            print(i)
        break
#
# # ************************************************************# ************************************************************
#
#             outputs = model(x)
#
#             outputs_logit = outputs.argmax(dim=1)
#
#             loss = criterion(outputs, y)
#             totalloss=totalloss+loss
#
#             num_iters += 1
#             loss.backward()
#
#             optimizer.step()
#             iteration += 1
#
#             # print("loss:" + str(loss.item()))
#             writer.add_scalar('Training/Loss', loss.item(), iteration)
#
#             predicts.append(outputs_logit.cpu().detach().numpy())
#             gts.append(y.cpu().detach().numpy())
#
#
#         predicts = np.concatenate(predicts).flatten().astype(np.int16)
#         gts = np.concatenate(gts).flatten().astype(np.int16)
#         loss1=totalloss/467
#         print("train_loss:" + str(loss1.item()))
#         x_axis.append(j)
#         t_loss.append(loss1)
#         print(metrics.confusion_matrix(predicts, gts))
#         acc = metrics.accuracy_score(predicts, gts)
#         recall = metrics.recall_score(predicts, gts)
#         f1 = metrics.f1_score(predicts, gts)
#
#         writer.add_scalar('Training/acc', acc, epoch)#
#         writer.add_scalar('Training/recall', recall, epoch)
#         writer.add_scalar('Training/f1', f1, epoch)
#         scheduler.step()
#
#
#         with torch.no_grad():
#             # val_loader = tqdm(val_loader)
#             for i, batch in enumerate(val_loader):
#                 model.eval()
#
#                 # print(f"Batch: {i}/{len(val_loader)} epoch {epoch}")
#
#                 x1 = batch['source']['data']
#                 y1 = batch['label']
#
#                 x1 = x1.type(torch.FloatTensor).cuda()
#                 y1 = y1.type(torch.LongTensor).cuda()
#
#
#                 outputs1 = model(x1)
#
#                 outputs1_logit = outputs1.argmax(dim=1)
#
#                 val_loss = criterion(outputs1, y1)
#                 val_totalloss = val_totalloss + val_loss
#
#                 predicts1.append(outputs1_logit.cpu().detach().numpy())
#                 gts1.append(y1.cpu().detach().numpy())
#
#             predicts1 = np.concatenate(predicts1).flatten().astype(np.int16)
#             gts1 = np.concatenate(gts1).flatten().astype(np.int16)
#             # acc2 = metrics.accuracy_score(predicts1, gts1)
#             # recall2 = metrics.recall_score(predicts1, gts1)
#             # f1_1 = metrics.f1_score(predicts1, gts1)
#
#             val_loss1 = val_totalloss/20
#             print("val_loss:" + str(val_loss1.item()))
#             v_loss.append(val_loss1)
#                 ## log
#             # print("val_acc:" + str(acc2))
#             # print("val_recall:" + str(recall2))
#             # print("val_f1:" + str(f1_1))
#             print(metrics.confusion_matrix(predicts1, gts1))
#
#
# # ************************************************************# ************************************************************
#             try:
#                 train_loss_lines.remove(train_loss_lines[0])  # 移除上一步曲线
#                 val_loss_lines.remove(val_loss_lines[0])
#             except Exception:
#                 pass
#             train_loss_lines = plt.plot(x_axis, t_loss, 'r', lw=0.5)  # lw为曲线宽度
#             val_loss_lines = plt.plot(x_axis, v_loss, 'b', lw=0.5)
#             plt.title("loss")
#             plt.xlabel("steps")
#             plt.ylabel("loss")
#             plt.legend(["train_loss", "val_loss"])
#             # plt.pause(0.1)  # 图片停留0.1s
#             if epoch % 30 == 0:
#                 # plt.savefig('testblueline.jpg')
#                 plt.savefig('loss' + '%d.jpg' % j)
# # ************************************************************# ************************************************************
#
# # ************************************************************# ************************************************************
#
# # ************************************************************# ************************************************************
#         scheduler.step()
#
#         # Store latest checkpoint in each epoch
#         torch.save(
#             {
#                 "model": model.state_dict(),
#                 "optim": optimizer.state_dict(),
#                 "scheduler": scheduler.state_dict(),
#                 "epoch": epoch,
#
#             },
#             os.path.join(r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\3dclassification\caihaihua log', 'checkpoint_latest.pt'),
#         )
# # ************************************************************# ************************************************************
#         # Save checkpoint
#         if epoch % 10 == 0:
#
#             torch.save(
#                 {
#                     "model": model.state_dict(),
#                     "optim": optimizer.state_dict(),
#                     "epoch": epoch,
#                 },
#                 os.path.join(r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\3dclassification\caihaihua log', f"checkpoint_{epoch:04d}.pt"),
#             )
#
#     writer.close()
# # ************************************************************# ************************************************************
train()