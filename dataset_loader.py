import torch
import torchvision
import torchvision.transforms as transforms


def load_image(batch_size, num_workers, random_state):
    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # データセットの読み込み
    train_set = torchvision.datasets.ImageFolder("./data/train", transform)
    valid_set = torchvision.datasets.ImageFolder("./data/valid", transform)

    # 乱数シードの固定
    g = torch.Generator()
    g.manual_seed(random_state)

    # DataLoaderに変換
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=g,
    )

    return train_loader, valid_loader
