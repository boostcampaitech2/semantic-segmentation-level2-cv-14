'''
이 파일에 Custom Dataset 클래스를 정의합니다.
 - DataSet_Trash : COCO format 으로 구성된 부스트캠프 재활용 분류 Segmentation 데이터를 로드합니다.
'''

import os
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as alb
from tqdm import tqdm
class DataSet_Trash(Dataset):
    def __init__(self, json_path, dataset_root, stage="train", loading_mode='preload', transforms=None):
        self.coco = COCO(json_path)
        self.dataset_root = dataset_root

        self.stage = stage

        loading_mode = loading_mode.lower()
        if loading_mode not in ['preload', 'realtime']:
            raise ValueError
        self.loading_mode = loading_mode

        self.datas = {}
        self.transforms = transforms

        # 이미지, Anntation 램에 미리 로드
        if self.loading_mode == 'preload':
            print("dataset preloading", flush=True)
            for index in tqdm(range(self.__len__())):
                self[index]

        self.mapper = {}


    '''
    카테고리를 인덱스 순서대로 반환합니다.
    '''
    def Categories(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        return ['Background'] + [c['name'] for c in categories]

    '''
    torch.Dataset 클래스의 인터페이스 구현입니다.
    콜백함수로, inference time 중에 인덱스에 해당하는 데이터셋을 호출합니다.
    인덱스에 해당하는 이미지와 이미지에 포함된 Annotation 들을 반환합니다.
    '''
    def __getitem__(self, index: int):
        if index == 2442: # 이미지 2442("batch_03/0702.jpg") 에 annotation 이 없어 예외처리
            return self.__getitem__(2441)

        if self.datas.get(index):
            data = self.datas[index]
        else:
            data = {}
            data["info"] = self.coco.loadImgs(index)[0]

            image = cv2.imread(os.path.join(self.dataset_root, data["info"]['file_name']))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data["image"] = image

            ann_ids = self.coco.getAnnIds(imgIds=data["info"]['id'])
            ann = self.coco.loadAnns(ann_ids)

            if self.stage in ["train", 'valid']:
                masks = np.zeros((data["info"]["height"], data["info"]["width"]))
                anns = sorted(ann, key=lambda idx: idx['area'], reverse=True)
                for i in range(len(anns)):
                    masks[self.coco.annToMask(anns[i]) == 1] = anns[i]['category_id']

                masks = masks.astype(np.int8)
                data["mask"] = masks

            if self.loading_mode == "preload":
                self.datas[index] = data

        if self.stage in ["train", 'valid']:
            transformed = self.transforms(image=data["image"], mask=data["mask"])
            return transformed["image"], transformed["mask"], data["image"]
        else:
            transformed = self.transforms(image=data["image"])
            return transformed["image"], data["info"]['file_name'], data["image"]

    '''
    torch.Dataset 클래스의 인터페이스 구현입니다.
    콜백함수로, inference time 중에 __getitem__ 함수에 feed 가능한 최대 인덱스 번호를 반환합니다.
    '''
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())
