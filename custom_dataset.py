import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        # csvファイルの読み込み
        data = pd.read_csv(csv_path)
        image_paths = data["path"]
        labels = data["label"]

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        # 画像読み込み
        path = self.image_paths[index]
        img = Image.open(path)

        # Transform事前処理
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        image_path = self.image_paths[index]
        return img, label, image_path

    def __len__(self):
        # データ数を返す
        return len(self.image_paths)
