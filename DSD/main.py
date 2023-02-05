# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from PIL import Image
import torch

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

class Steel_Data(torch.utils.data.Dataset):
    def __init__(self, csv_file, mode='train', transform=None):
        
        self.mode = mode # 'train', 'val' or 'test'
        self.data_list = []
        self.category = []
        self.transform = transform
        
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile) #用dictionary的方式讀取csv的資料
            for row in reader:
                self.data_list.append(row['ImageId']) #將key 為 file_path的value讀進data_list
                if mode != 'test':
                    self.category.append(int(row['ClassId'])-1)
        if mode == 'train':
            self.data_list = self.data_list[0:6000]
            self.category = self.category[0:6000]
        if mode == 'val':
            self.data_list = self.data_list[6000:7096]
            self.category = self.category[6000:7096] 

    def __getitem__(self, index):

        data = Image.open('../input/severstal-steel-defect-detection/train_images/'+ self.data_list[index])
        if self.transform is not None:
            data = self.transform(data)
        if self.mode == 'test':
            return data
        category = torch.tensor(int(self.category[index]))
        return data, category

    def __len__(self):
        return len(self.data_list)
    
from torchvision import transforms
# For TRAIN
########################################################################
#  TODO: use transforms.xxx method to do some data augmentation        #
#  This one is for training, find the composition to get better result #
########################################################################
transforms_train = transforms.Compose([
transforms.Resize((256, 256)),         #將照片固定為196x196的大小
transforms.RandomCrop((224, 224)),      #將照片隨機裁減為224x224的大小
transforms.RandomHorizontalFlip(p=0.5), #0.5的機率是否水平翻轉
transforms.RandomVerticalFlip(p=0.5),   #0.5的機率是否垂直翻轉
transforms.RandomRotation(degrees=(-90, 90)),  #隨機地在-90~90度間旋轉
transforms.ToTensor(),  #將照片轉成tensor 並且將數值都轉換成0~1 
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) #標準化 
########################################################################
#                           End of your code                           #
########################################################################

# For VAL, TEST
########################################################################
#  TODO: use transforms.xxx method to do some data augmentation        #
#  This one is for validate and test,                                  #
#  NOTICE some operation we usually not use in this part               #
########################################################################
transforms_test = transforms.Compose([
transforms.Resize((256, 256)),
transforms.CenterCrop((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
########################################################################
#                           End of your code                           #
########################################################################

dataset_train = Steel_Data('../input/severstal-steel-defect-detection/train.csv', mode='train',transform=transforms_train)
dataset_val = Steel_Data('../input/severstal-steel-defect-detection/train.csv', mode='val', transform=transforms_train)

print("The first image's shape in dataset_train :", dataset_train.__getitem__(0)[0].size()) #[0]means data
print("There are", dataset_train.__len__(), "images in dataset_train.")
print('-'*50)

print("The first image's shape in dataset_val :", dataset_val.__getitem__(0)[0].size()) #[0]means data
print("There are", dataset_val.__len__(), "images in dataset_val.")

# 224x224 because of transformation
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False)

import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.models as models
resnet101=torchvision.models.resnet101(pretrained=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.cnn_model = resnet101
        self.fc1 = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(p=0.4),
            nn.Linear(500, 4))
    def forward(self, x):
        x = self.cnn_model(x)
        out = self.fc1(x)
        return out
    
model = Net()
device = torch.device('cuda:0')
model = model.to(device)

import torch.nn as nn
import torch.optim as optim
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss function combines both a SoftMax activation 
                                  # and a cross entropy loss function in the same function 
                                  # 這正是為什麼我們沒有在最後用 SoftMax轉換的原因
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,momentum=0.9)
criterion = criterion.to(device)
def train(input_data, model, criterion, optimizer):
    '''
    Argement:
    input_data -- iterable data, typr torch.utils.data.Dataloader is prefer
    model -- nn.Module, model contain forward to predict output
    criterion -- loss function, used to evaluate goodness of model
    optimizer -- optmizer function, method for weight updating
    '''
    model.train()
    total_count = 0
    acc_count = 0
    total_run = 0
    total_f1_score = 0
    for i, data in enumerate(input_data, 0):
        images, categorys = data[0].to(device), data[1].to(device)
        
        ########################################################################
        # TODO: Forward, backward and optimize                                 #
        # 1. zero the parameter gradients                                      #
        # 2. process input through the network                                 #
        # 3. compute the loss                                                  #
        # 4. propagate gradients back into the network’s parameters            #
        # 5. Update the weights of the network                                 #
        ########################################################################
        # Run the forward 
        outputs = model(images)
        loss = criterion(outputs, categorys) 

        # Backward and perform optimization
        optimizer.zero_grad() #將梯度初始化為0，這步很關鍵，因為每次我們使用的batch不同，導致loss不同，因此梯度函數不同
        loss.backward() #進行反向傳播
        optimizer.step() #藉由反向傳播的結果計算梯度
        ########################################################################
        #                           End of your code                           #
        ########################################################################


        ########################################################################
        # TODO: Get the counts of correctly classified images                  #
        # 1. get the model predicted result                                    #
        # 2. sum the number of this batch predicted images                     #
        # 3. sum the number of correctly classified                            #
        # 4. save this batch's loss into loss_list                             #
        # dimension of outputs: [batch_size, number of classes]                #
        # Hint 1: use outputs.data to get no auto_grad                         #
        # Hint 2: use torch.max()                                              #
        ########################################################################
        _, predicted = torch.max(outputs.data,1) #返回每一行中最大值的那个元素，且返回其索引
        total_count += categorys.size(0) #x.size(0)指的是batch size (目前設定為32)
        acc_count += (predicted == categorys).sum().item() #分類正確的總數量
        total_run += 1
        ########################################################################
        #                           End of your code                           #
        ########################################################################

    # Compute this epoch accuracy and loss
    acc = acc_count / total_count

    return acc

def val(input_data, model, criterion, optimizer):
    model.eval()
    
    total_count = 0
    acc_count = 0
    total_run = 0
    total_f1_score = 0
    with torch.no_grad():
        for data in input_data:
            images, categorys = data[0].to(device), data[1].to(device)

            ####################################################################
            # TODO: Get the predicted result and loss                          #
            # 1. process input through the network                             #
            # 2. compute the loss                                              #
            # 3. get the model predicted result                                #
            # 4. get the counts of correctly classified images                 #
            # 5. save this batch's loss into loss_list                         #
            ####################################################################
            outputs_val = model(images)
            loss = criterion(outputs_val, categorys)

            _, predicted = torch.max(outputs_val.data,1)
            total_count += categorys.size(0)
            acc_count += (predicted == categorys).sum().item()
            total_run += 1
            ####################################################################
            #                         End of your code                         #
            ####################################################################

    acc = acc_count / total_count
    return acc

################################################################################
# You can adjust those hyper parameters to loop for max_epochs times           #
################################################################################
max_epochs = 50
log_interval = 1 # print acc and loss in per log_interval time
################################################################################
#                               End of your code                               #
################################################################################
train_acc_list = []
val_acc_list = []

for epoch in range(1, max_epochs + 1):
    print('=' * 20, 'Epoch', epoch, '=' * 20)
    train_acc = train(train_loader, model, criterion, optimizer)
    val_acc = val(val_loader, model, criterion, optimizer)

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    if epoch % log_interval == 0:
        print('Train Acc: {:.6f}'.format(train_acc))
        print('Val Acc: {:.6f}'.format(val_acc))