import os
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def test_data(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test/dnd/')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'noise/'))
        self._check_image(self.image_list)
        self.image_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'noise', self.image_list[idx]))
        image = F.to_tensor(image)
        name = self.image_list[idx]
        return image, name

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['PNG', 'jpg', 'jpeg', 'bmp', 'JPG', 'png']:
                raise ValueError

