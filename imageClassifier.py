import tensorflow as tf
import keras
from keras import layers, regularizers, optimizers
from keras.layers import Dense, Add, Flatten, Input, Activation, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Lambda, concatenate
from keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from numpy import load, save
import re
import keras.backend as K
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from torch.utils import data
import torch.optim as optim
from torchvision.transforms.functional import pad
from torchvision import transforms
import numpy as np
import numbers
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import WeightedRandomSampler

def predictClass(imgpath):
    PATH = './imgModel/final_model_13'

    pretrained = {
      'resnet152':torch.hub.load('pytorch/vision:v0.6.0', 'resnet152', pretrained=True),
      'vgg16':torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True),
      'inceptionv3': torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True,aux_logits=False)

    }
    num_classes = 13


    class Resnet152(nn.Module):
        def __init__(self):
            super(Resnet152, self).__init__()

            # Everything except the last linear layer
            self.model = nn.Sequential(*list(pretrained['resnet152'].children())[:-1])

            # now to freeze [:-2] layers
            child_counter = 0
            cc = 0

            for child in self.model.children():

                if child_counter < 7:

                    for param in child.parameters():
                        param.requires_grad = False

                elif child_counter == 7:
                    for child1 in child.children():
                        for param in child1.parameters():
                            if cc < 2:
                                param.requires_grad = False
                            cc = cc + 1
                child_counter += 1

            # custom fc-layer
            self.classifier_layer = nn.Sequential(

                nn.Linear(2048, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Linear(1024, num_classes)
            )

        def forward(self, x):
            f = self.model(x)
            # print(f.size())
            f = f.view(-1, 2048)
            # print(f.size())
            y = self.classifier_layer(f)
            return y

    device = 'cuda'
    rs = Resnet152().cuda()
    model = rs
    model.load_state_dict(torch.load(PATH))
    model.eval()

    path = './data/Image_Data_Split'
    path_val = './data/Image_Data_Split/test'
    path_predict = ''

    maps = ['SF', 'CS', 'SL', 'Keypad', 'Samsung', 'CB', 'Apple', 'C(BLUE)', 'C(yellow)','C(RG)', 'MOTO','huawei', 'C(Glacier White)']
    id2idx = {}
    for i, m in enumerate(maps):
        id2idx[m] = i

    maps = ['SF', 'CS', 'SL', 'Keypad', 'Samsung', 'CB', 'Apple', 'C(BLUE)', 'C(yellow)','C(RG)', 'MOTO','huawei', 'C(Glacier White)']
    idx2id = {}
    for i, m in enumerate(maps):
        idx2id[i] = m
    # print(idx2id)

    def img_load(dir):
        labels = {}
        c = 0
        w = 0
        h = 0
        for f in os.listdir(path):
          if f == 'test' or f == 'predict':
            continue
          #if f != 'Check':
          for tags in os.listdir(path+'/'+f):
              if tags == 'Apple':
                tags = tags
              if tags == 'SF':
                tags = tags
              for img in os.listdir(path+'/'+f+'/'+tags):
                labels[f+'/'+tags+'/'+img] = tags.split('/')[0]
        for i,v in labels.items():
          labels[i] = id2idx[v]
        return labels


    def img_val(dir):
        val_labels = {}

        for path in os.listdir(dir):
            for img in os.listdir(dir + '/' + path):
                if img == '.ipynb_checkpoints':
                    continue
                val_labels[path + '/' + img] = path

        for i, v in val_labels.items():
            val_labels[i] = id2idx[v]
        return val_labels


    def img_predict(dir):
        val_labels = {}

        val_labels[imgpath] = dir

        # for img in os.listdir(dir):
        #     val_labels[dir + '/' + img] = dir

        for i, v in val_labels.items():
            val_labels[i] = 0

        return val_labels


    class CustomDataSet(Dataset):
        def __init__(self, main_dir, transform, var):
            self.main_dir = main_dir
            self.transform = transform

            self.dic = {}
            if var == 'train':
                self.dic = img_load(main_dir)
            elif var == 'test':
                self.dic = img_val(main_dir)
            elif var == 'predict':
                self.dic = img_predict(main_dir)

            self.total_imgs = list(self.dic.keys())

        def __len__(self):
            return len(self.total_imgs)

        def __getitem__(self, idx):
            o = self.total_imgs[idx]
            img_loc = os.path.join(self.main_dir, o)
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
            tags = self.dic[o]
            return tensor_image, tags

    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = np.max([w, h])
            hp = int((max_wh - w) / 2)
            vp = int((max_wh - h) / 2)
            padding = (hp, vp, hp, vp)
            return pad(image, padding, 0, 'constant')

    image_size = (224,224)

    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

    ])

    f = img_load(path)
    k = f.values()
    # print(k)

    k = {i:0 for i in range(len(maps))}
    for key,value in f.items():
        k[value] = k[value]+1

    class_count = [i for i in list(k.values())]
    class_weights_criterion = 1./torch.tensor(class_count, dtype=torch.long)
    # print(class_weights_criterion)

    class_weights = [len(f)/class_count[i] for i in range(len(class_count))]
    weights = [class_weights[list(f.values())[i]] for i in range(len(f))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(f))

    my_dataset_val = CustomDataSet(path_val,transform, 'test')
    val_loader = data.DataLoader(my_dataset_val , batch_size=1, shuffle=True)

    my_dataset_predict = CustomDataSet(path_predict, transform, 'predict')
    pred_loader = data.DataLoader(my_dataset_predict, batch_size=1, shuffle=True)

    def imshow(img):
        plt.figure(figsize=(20,20))
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # plt.plot(figsize=(5,4))
    y_pred_list = []
    y_test = []
    with torch.no_grad():
        for inputs,classes in pred_loader:
            inputs = inputs.to(device)
            y_test.append(classes)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred_list.append(preds.cpu().numpy())
            # if preds != classes :
              # imshow(torchvision.utils.make_grid(inputs.cpu()))
              # print(idx2id[preds.cpu().numpy()[0]],idx2id[classes.cpu().numpy()[0]])


    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_test = [b.squeeze().tolist() for b in y_test]

    plt.plot(figsize=(5,4))
    y_pred_list = []
    y_test = []
    pred_val = ""
    with torch.no_grad():
        for inputs,classes in val_loader:
            inputs = inputs.to(device)
            y_test.append(classes)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred_list.append(preds.cpu().numpy())
            if preds != classes :
              if classes.cpu().numpy()[0] == id2idx['huawei']:
                # imshow(torchvision.utils.make_grid(inputs.cpu()))
                pred_val = idx2id[preds.cpu().numpy()[0]]
                # print(idx2id[preds.cpu().numpy()[0]],idx2id[classes.cpu().numpy()[0]])


    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_test = [b.squeeze().tolist() for b in y_test]

    # print(classification_report(y_test,y_pred_list))

    return pred_val


