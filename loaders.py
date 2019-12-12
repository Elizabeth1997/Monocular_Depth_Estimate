import glob
import torch
import numpy as np
from skimage import io, transform
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

np.random.seed(0) 
torch.manual_seed(0)
VALIDATION_SPLIT = 0.02

class DepthHalfSize(object):
    def __call__(self, sample):
        x = sample['depth']
        sample['depth'] = transform.resize(x, (x.shape[0]//2, x.shape[1]//2))
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        # swap channel axis
        image = image.transpose((2, 0, 1))
        depth = depth.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'depth': torch.from_numpy(depth)}
    
class DepthToNormal(object):
    def __call__(self, sample):
        dx, dy = np.gradient(sample['depth'].squeeze())
        dx, dy, dz = dx * 2500, dy * 2500, np.ones_like(dy)
        n = np.linalg.norm(np.stack((dy, dx, dz), axis=-1), axis=-1)
        d = np.stack((dy/n, dx/n, dz/n), axis=-1)
        return {'image': sample['image'], 'depth': (d + 1) * 0.5} 
        
class ImageDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform 
        self.image_files = glob.glob(root_dir + '/*.jpg')
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = io.imread(self.image_files[idx]) / 255.0
        depth = io.imread(self.image_files[idx].replace('.jpg', '.png'))[:,:,:1] / 255.0        
        sample = {'image': image, 'depth': depth}        
        return self.transform(sample) if self.transform else sample
    
def prep_loaders(root_dir=None, batch_size=1, workers=1):
    # Load dataset
    image_depth_dataset = ImageDepthDataset(root_dir=root_dir, transform=transforms.Compose([DepthHalfSize(), ToTensor()]))

    # Split into training and validation sets
    train_size = int((1-VALIDATION_SPLIT) * len(image_depth_dataset))
    test_size = len(image_depth_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(image_depth_dataset, [train_size, test_size])

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    print('Dataset size (num. batches)', len(train_loader), len(valid_loader))
    
    return train_loader, valid_loader

if __name__ == '__main__':
    pass