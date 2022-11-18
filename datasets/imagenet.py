import os.path as osp
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset

class ImageNet(Dataset):

    def __init__(
        self,
        phase='train',
        trans=None
    ) -> None:
        root_dir = './data/ImageNet'
        map_path = osp.join(root_dir, 'LOC_synset_mapping.txt')
        img_dir = osp.join(root_dir, 'ILSVRC', 'Data', 'CLS-LOC', phase)
        anno_dir = osp.join(root_dir, 'ILSVRC', 'Annotations', 'CLS-LOC', phase)
        self.trans = trans

        with open(map_path, 'r') as f:
            lines = f.readlines()
        # key: wnid, value: class name
        cls_map = {}
        for line in lines:
            line = line.rstrip().split(' ')
            key = line[0]
            val = ' '.join(line[1:]).split(',')
            val = val[0][:-1] if val[0][-1] == ',' else val[0]
            cls_map[key] = val

        # get images and class labels.
        self.imgs = sorted(glob.glob(osp.join(img_dir, '**', '*.JPEG'), recursive=True))
        self.labels = []
        if phase == 'train':
            for img in self.imgs:
                label = cls_map[osp.basename(img).split('_')[0]]
                self.labels.append(label)
        elif phase == 'val':
            for img in self.imgs:
                anno = osp.join(anno_dir, osp.basename(img).split('.')[0]+'.xml')
                tree = ET.parse(anno)
                root = tree.getroot()
                self.labels.append(cls_map[root[5][0].text])
        else:
            raise Exception

        le = preprocessing.LabelEncoder()
        self.labels = torch.as_tensor(le.fit_transform(self.labels))        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # image
        with Image.open(self.imgs[index]) as img:
            img = img.convert('RGB')
            if self.trans:
                img = self.trans(img)
        # label
        label = self.labels[index]
        return img, label 


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Resize, Normalize
    trans = Compose([Resize((224, 224)), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = ImageNet(trans=trans)
    dataloader = DataLoader(dataset, shuffle=True)
    img, label = next(iter(dataloader))
    print(img.size())
    print(label)