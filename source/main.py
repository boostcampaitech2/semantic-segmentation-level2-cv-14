import argparse
import os
import numpy as np
import random
import pandas as pd

from trainer import trainer
from inference import tester
from transforms import train_transform,val_transform,test_transform
from dataset import CustomDataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

import segmentation_models_pytorch as smp
import wandb

'''
python main.py --mode=train --num_epochs=100 --model_name=pspnet_efficientb0_mIoUbest
python main.py --mode=test --model_name=pspnet_efficientb0_mIoUbest --saved_epoch=92
'''

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(mode,saved_dir,num_epochs,batch_size,learning_rate,model_name,saved_epoch):

    #plt.rcParams['axes.grid'] = False

    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train.json / validation.json / test.json 디렉토리 설정
    dataset_path = '../input/data'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # train dataset
    category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
                    'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    train_dataset = CustomDataLoader(data_dir=train_path, category_names=category_names, dataset_path=dataset_path, mode='train', transform=train_transform)
    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path, category_names=category_names, dataset_path=dataset_path, mode='val', transform=val_transform)
    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_path, category_names=category_names, dataset_path=dataset_path, mode='test', transform=test_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn)
    
    # model 선언
    #model = models.segmentation.fcn_resnet50(pretrained=True)
    #model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
    model = smp.PSPNet(
        encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=11,                      # model output channels (number of classes in your dataset)
    )
    
    
    if mode=='train':
        #for wandb
        wandb.watch(
            model, criterion=None, log="gradients", log_freq=100, idx=None,
            log_graph=(False)
        )

        # 모델 저장 함수 정의
        val_every = 1
        if not os.path.isdir(saved_dir):                                                           
            os.mkdir(saved_dir)

        # Loss function 정의
        criterion = nn.CrossEntropyLoss()
        # Optimizer 정의
        optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate, weight_decay=1e-6)

        Trainer = trainer(saved_dir, device, category_names)
        saved_epoch = Trainer.train(num_epochs,model,train_loader,val_loader,val_every,criterion,optimizer,model_name)
        print(f'Training Complete! saved_epoch={saved_epoch}')

    elif mode=='test':
        # sample_submisson.csv 열기
        submission = pd.read_csv('../submission/sample_submission.csv', index_col=None)

        #모델 불러오기
        model_path = os.path.join(saved_dir, model_name+'_epoch'+saved_epoch+'.pt')
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.state_dict()
        model.load_state_dict(state_dict)
        model = model.to(device)
        #model.eval()

        # test set에 대한 prediction
        file_names, preds = tester(device).test(model,test_loader)

        # PredictionString 대입
        for file_name, string in zip(file_names, preds):
            submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                        ignore_index=True)

        # submission.csv로 저장
        submission.to_csv(f'../submission/{model_name}_epoch{saved_epoch}.csv', index=False)
        print('Prediction result saved!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='.pth file name') 
    parser.add_argument('--saved_dir', type=str, default='./saved',help='path of train configuration yaml file')
    parser.add_argument('--num_epochs', type=int, default=50, help='path of inference configuration yaml file') 
    parser.add_argument('--batch_size', type=int, default=16, help='path of inference configuration yaml file') 
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='path of inference configuration yaml file') 
    parser.add_argument('--seed', type=int, default=21, help='path of inference configuration yaml file') 
    parser.add_argument('--model_name', type=str, default='fcn_resnet50_best_model(pretrained)', help='.pth file name') 
    parser.add_argument('--saved_epoch', type=str, default='0', help='.pth file name') 
    args = parser.parse_args()

    #wandb 설정
    if args.mode=='train':
        wandb.init(project=args.model_name,config=args)
        #wandb.config.update(args)


    #seed 고정
    seed_everything(args.seed)

    #train (or test)
    train(args.mode,args.saved_dir,args.num_epochs,args.batch_size,args.learning_rate,args.model_name,args.saved_epoch)

    
