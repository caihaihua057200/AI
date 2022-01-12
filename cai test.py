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
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# **************************************************# ************************************************************
# 文件目录传输
source_train_0_dir = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\source_train_0_dir'
source_train_1_dir = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\source_train_1_dir'
source_test_0_dir = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\source_test_0_dir'
source_test_1_dir = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\source_test_1_dir'


output_dir= r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\caihaihua log'

latest_checkpoint_file = 'checkpoint_0050.pt'



from data_function import MedData_test
from models.three_d.densenet3dcai import generate_model
model = generate_model(121, n_input_channels=1, num_classes=2, drop_rate=0)

# from models.two_d.googlenet import googlenet
# model = googlenet(num_class=2)
# from models.three_d.densenet3d import generate_model
# model = generate_model(264, n_input_channels=1, num_classes=2, drop_rate=0.)

model = torch.nn.DataParallel(model, device_ids=devicess, output_device=[1])
model.eval()

# print("load model:")
# print(os.path.join(hp.output_dir, hp.latest_checkpoint_file))
ckpt = torch.load(os.path.join(output_dir, latest_checkpoint_file),
                  map_location=lambda storage, loc: storage)

model.load_state_dict(ckpt["model"])

model.cuda()

test_dataset = MedData_test(source_test_0_dir, source_test_1_dir)
test_loader = DataLoader(test_dataset.testing_set,
                         batch_size=1,
                         shuffle=True,
                         pin_memory=True,
                         drop_last=True)

model.eval()

predicts = []
gts = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        model.eval()

        x = batch['source']['data']
        y = batch['label']

        x = x.type(torch.FloatTensor).cuda()
        y = y.type(torch.LongTensor).cuda()


        # print(x.shape)




        outputs = model(x)
        # print(outputs)


        outputs_logit = outputs.argmax(dim=1)
        # print(outputs_logit, y)

        predicts.append(outputs_logit.cpu().detach().numpy())
        gts.append(y.cpu().detach().numpy())





predicts = np.concatenate(predicts).flatten().astype(np.int16)
gts = np.concatenate(gts).flatten().astype(np.int16)
# print(gts)
# print(predicts)
acc = metrics.accuracy_score(predicts, gts)
recall = metrics.recall_score(predicts, gts)
f1 = metrics.f1_score(predicts, gts)
## log
print("acc:" + str(acc))
print("recall:" + str(recall))
print("f1:" + str(f1))
print(metrics.confusion_matrix(predicts, gts))





