import albumentations as A
import numpy as np
import torch
#import tqdm
import wandb

class tester:
    def __init__(self,device):
        self.device=device

    def test(self,model,test_loader):
        size = 256
        transform = A.Compose([A.Resize(size, size)])
        print('Start prediction.')
        
        model.eval()
        
        file_name_list = []
        preds_array = np.empty((0, size*size), dtype=np.long)
        
        with torch.no_grad():
            for step, (imgs, image_infos) in enumerate(test_loader):
                
                # inference (512 x 512)
                outs = model(torch.stack(imgs).to(self.device))#['out']
                oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
                
                # resize (256 x 256)
                temp_mask = []
                for img, mask in zip(np.stack(imgs), oms):
                    transformed = transform(image=img, mask=mask)
                    mask = transformed['mask']
                    temp_mask.append(mask)
                    
                oms = np.array(temp_mask)
                
                oms = oms.reshape([oms.shape[0], size*size]).astype(int)
                preds_array = np.vstack((preds_array, oms))
                
                file_name_list.append([i['file_name'] for i in image_infos])

        print("End prediction.")
        file_names = [y for x in file_name_list for y in x]
        
        return file_names, preds_array
