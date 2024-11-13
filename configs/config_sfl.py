from torch.optim import Adam, lr_scheduler
from torchvision.transforms import transforms


def get_optimizer_and_scheduler(model):
    optimizer = Adam(model.parameters(), lr=4e-5, weight_decay=1e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.955)
    return optimizer, scheduler


def get_transforms():
    return transforms.Compose([
        transforms.GaussianBlur(3),
        transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-0.3, 0.3]),
        transforms.ToTensor(),
    ])


config = {
    'TOTAL_EPOCHS': 50,
    'DISTANCE_WEIGHT': 1,  # The parameter lambda in the paper
    'DISTANCE_START_EPOCH': 5,  # Should be set to an epoch when model starts predicting normally again
}
