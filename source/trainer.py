import os
import random
import configparser
from importlib import import_module

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

from utils import save_model,label_accuracy_score, add_hist
import wandb

class trainer:
    def __init__(self, saved_dir, device, category_names):
        self.saved_dir=saved_dir
        self.device=device
        self.category_names=category_names
        temp_dict={}
        for i in range(len(category_names)):
            temp_dict[i]=category_names[i]
        self.category_dict=temp_dict
        print(self.category_dict)

    def validation(self, epoch, model, data_loader, criterion, device):
        print(f'Start validation #{epoch}')
        model.eval()

        with torch.no_grad():
            n_class = 11
            total_loss = 0
            cnt = 0
            
            hist = np.zeros((n_class, n_class))
            example_images = []#for wandb
            for step, (images, masks, _) in enumerate(data_loader):
                
                images = torch.stack(images)       
                masks = torch.stack(masks).long()  

                images, masks = images.to(device), masks.to(device)            
                
                # device 할당
                model = model.to(device)
                
                outputs = model(images)#['out']
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)

                #for wandb
                example_images.append(wandb.Image(
                    images[0],
                    masks={
                        "predictions" : {
                            "mask_data" : outputs[0],
                            "class_labels" : self.category_dict
                        },
                        "ground_truth" : {
                            "mask_data" : masks[0],
                            "class_labels" : self.category_dict
                        }
                    },
                    caption=f'epoch:{epoch} step:{step}'
                ))

            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            IoU_by_class = {classes : round(IoU,4) for IoU, classes in zip(IoU , self.category_names)}
            
            avrg_loss = total_loss / cnt
            wandb.log({
                'valid/avg_loss':round(avrg_loss.item(),4), 
                'valid/acc' :round(acc, 4), 
                'valid/classification_acc':acc_cls,
                'valid/mIoU': round(mIoU,4),
                'IoU_by_class':IoU_by_class,
                'examples':example_images,})

    
            print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                    mIoU: {round(mIoU, 4)}')
            print(f'IoU by class : {IoU_by_class}')
            
        return mIoU
    

    def train(self,num_epochs,model,train_loader,val_loader,val_every,criterion,optimizer,model_name):
        print(f'Start training..')
        n_class = 11
        best_mIoU = -1
        saved_epoch = -1

        for epoch in range(num_epochs):
            model.train()

            hist = np.zeros((n_class, n_class))
            for step, (images, masks, _) in enumerate(train_loader):
                images = torch.stack(images)       
                masks = torch.stack(masks).long() 
                
                # gpu 연산을 위해 device 할당
                images, masks = images.to(self.device), masks.to(self.device)
                
                # device 할당
                model = model.to(self.device)
                
                # inference
                outputs = model(images)#['out']
                
                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_class)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

                # step 주기에 따른 loss 출력
                if (step + 1) % 25 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                            Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                    wandb.log({
                        'train/loss':round(loss.item(),4),
                        'train/acc': acc, 
                        'train/classification_acc':acc_cls,
                        'train/mIoU': round(mIoU,4),
                        'config/lr': optimizer.param_groups[0]["lr"]
                        })
                
            # validation 주기에 따른 loss 출력 및 best model 저장
            if (epoch + 1) % val_every == 0:
                mIoU = self.validation(epoch + 1, model, val_loader, criterion, self.device)
                if mIoU > best_mIoU: #mIou로 바꾸기
                    print(f"Best performance at epoch: {epoch + 1}")
                    print(f"Save model in {self.saved_dir}")
                    best_mIoU = mIoU
                    save_model(model, self.saved_dir,model_name+'_epoch'+str(epoch+1)+'.pt')
                    saved_epoch=epoch+1

        return saved_epoch