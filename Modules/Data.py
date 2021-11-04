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
import random
from Modules.Transforms_Preprocess import ObjectAugmentation

class DataSet_Trash(Dataset):
    def __init__(self, json_path, dataset_root, stage, loading_mode='preload', transforms=None, object_aug=True):
        self.coco = COCO(json_path)
        self.dataset_root = dataset_root

        self.stage = stage

        loading_mode = loading_mode.lower()
        if loading_mode not in ['preload', 'realtime']:
            raise ValueError
        self.loading_mode = loading_mode

        self.datas = {}
        self.transforms = transforms

        self.len = len(self.coco.getImgIds())
        self.object_aug = object_aug

        # 이미지, Anntation 램에 미리 로드
        if self.loading_mode == 'preload':
            print("dataset preloading", flush=True)
            for index in tqdm(range(self.__len__())):
                self[index]

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

                masks = masks.astype(np.uint8)
                data["mask"] = masks

            if self.loading_mode == "preload":
                self.datas[index] = data

        if self.stage == "train" and self.object_aug and random.random() > 0.5:
            target_image, target_mask = data["image"].copy(), data["mask"].copy()
            source_image, source_mask = self.getRandomInterestData()

            # 마스크 onehot 형식으로 변환해 클래스별로 차원을 나눔
            masks_source_onehot = (np.arange(11) == source_mask[..., None]).transpose(2, 0, 1).astype(np.uint8)
            masks_target_onehot = (np.arange(11) == target_mask[..., None]).transpose(2, 0, 1).astype(np.uint8)

            # onehot 형식으로 변환된 마스크를 클래스별로 순회하며 원하는 클래스만 합성
            for idx in range(11):
                if idx not in [1, 3, 4, 5, 6, 10]:
                    continue
                    
                # 다른 segmentation과 겹치지 않도록 덮어씌워질 부분을 0으로 만듦
                for idx_remove in range(11):
                    masks_target_onehot[idx_remove][masks_source_onehot[idx] > 0] = 0
                    
                target_image[masks_source_onehot[idx] > 0] = source_image[masks_source_onehot[idx] > 0] # 이미지 합성
                masks_target_onehot[idx][masks_source_onehot[idx] > 0] = masks_source_onehot[idx][masks_source_onehot[idx] > 0] # 마스크 합성

            # 합성된 마스크 다시 인덱스형식으로 변환
            target_mask = np.zeros((512, 512), dtype=np.uint8)
            for i in range(11):
                target_mask[masks_target_onehot[i] == 1] = i

            transformed = self.transforms(image=target_image, mask=target_mask)
            return transformed["image"], transformed["mask"], data["image"]

        if self.stage in ["train", 'valid']:
            transformed = self.transforms(image=data["image"], mask=data["mask"])
            return transformed["image"], transformed["mask"], data["image"]
        else:
            transformed = self.transforms(image=data["image"])
            return transformed["image"], data["info"]['file_name'], data["image"], data["info"]["id"]

    '''
    torch.Dataset 클래스의 인터페이스 구현입니다.
    콜백함수로, inference time 중에 __getitem__ 함수에 feed 가능한 최대 인덱스 번호를 반환합니다.
    '''
    def __len__(self) -> int:
        return self.len

    def getRandomInterestData(self):
        data = {}
        while True:
            index = random.randrange(0, self.len)
            data["info"] = self.coco.loadImgs(index)[0]

            ann_ids = self.coco.getAnnIds(imgIds=data["info"]['id'])
            anns = self.coco.loadAnns(ann_ids)

            categorys = set([ann['category_id'] for ann in anns])

            if len(categorys & {1, 3, 4, 5, 6, 10}) > 0:
                break

        image = cv2.imread(os.path.join(self.dataset_root, data["info"]['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data["image"] = image

        ann_ids = self.coco.getAnnIds(imgIds=data["info"]['id'])
        ann = self.coco.loadAnns(ann_ids)

        masks = np.zeros((data["info"]["height"], data["info"]["width"]))
        anns = sorted(ann, key=lambda idx: idx['area'], reverse=True)
        for i in range(len(anns)):
            masks[self.coco.annToMask(anns[i]) == 1] = anns[i]['category_id']

        masks = masks.astype(np.int8)

        result = ObjectAugmentation()(image=image, mask=masks)
        return result["image"], result["mask"]
