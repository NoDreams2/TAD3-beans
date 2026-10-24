"""Загрузка картинок для обучения и оценки."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# средние и разброс цветов ImageNet: на них обучались предобученные модели
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def make_loader(folder, batch_size: int, num_workers: int, shuffle: bool, generator=None):
    ds = datasets.ImageFolder(str(folder), transform=TRANSFORM)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, generator=generator)
