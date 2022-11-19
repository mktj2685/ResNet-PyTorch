import os
import os.path as osp
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class Caltech256(Dataset):

    def __init__(self, phase:str, transforms=None):
        data_dir = 'data/Caltech256/256_ObjectCategories'
        image_dirs = os.listdir(data_dir)
        image_dirs.remove('257.clutter')
        self.image_paths = []
        self.trans = transforms
        for image_dir in image_dirs:
            image_paths = glob(osp.join(data_dir, image_dir, '*.jpg'))
            if phase == 'train':
                self.image_paths += image_paths[20:]
            elif phase == 'val':
                self.image_paths += image_paths[:20]
            else:
                raise Exception

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = int(osp.basename(image_path).split('_')[0]) - 1   # label start at 0.
        with Image.open(image_path) as image:
            image = image.convert('RGB')    # grayscale to rgb
            if self.trans:
                image = self.trans(image)
        return image, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Resize, Normalize

    trans = Compose([Resize((224, 224)), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = Caltech256('val', trans)
    dataloader = DataLoader(dataset)

    img, label = next(iter(dataloader))
    print(img.size())