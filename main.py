import os
import torch
import torch.optim as optim
import random
import numpy as np
import wandb
from torch import nn
from tqdm import tqdm
from datetime import datetime

from simple_cnn import SimpleCNN
import validation_functions as valid
import dataset_loader as loader

num_epochs = 10  # 学習の回数
batch_size = 5  # 一回の学習でデータ取り込む数
num_workers = 2  # 並列実行の数

# 乱数シードの固定
random_state = 111
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)


def train():
    # 学習モデルの出力先フォルダを作成
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"outputs/{now}"
    results = ""
    os.makedirs(output_dir)

    # W&Bのログを設定
    wandb.init(project="DeepLearningSample")

    # CUDA（GPU）を使用するように切り替える
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.multiprocessing.freeze_support()

    # CNNモデルを指定
    model = SimpleCNN()
    model = model.to(device)

    # 損失関数（クロスエントロピー）、最適化関数（Adam）を設定
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 学習・検証データの読み込み
    train_loader, valid_loader = loader.load_image(batch_size, num_workers, random_state)

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        print(f"Epoch: {epoch}")

        # 学習フェーズ
        model = model.train()
        for i, data in tqdm(enumerate(train_loader, 0)):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # モデルを使用して推論する
            optimizer.zero_grad()
            outputs = model(inputs)

            # 損失・勾配を計算してモデルを更新
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 検証フェーズ
        model = model.eval()
        with torch.no_grad():
            running_score = 0.0

            for j, data in tqdm(enumerate(valid_loader, 0)):
                inputs = data[0].to(device)
                labels = data[1].to(device)

                # モデルの精度（正解率）を検証する
                pred = model(inputs)
                running_score += valid.get_classification_accuracy(pred, labels)

        # 各エポック結果を記録
        epoch_loss, epoch_score = running_loss / (i + 1), running_score / (j + 1)
        wandb.log({"Epoch": epoch, "Loss": epoch_loss, "Score": epoch_score})
        results += ("Epoch:" + str(epoch) + "  " + f"Loss: {epoch_loss}  Score: {epoch_score}\n")

        print(f"Loss: {epoch_loss}  Score: {epoch_score}\n")

    # モデル学習完了後の保存
    torch.save(model.state_dict(), output_dir + "/model.pth")
    with open(output_dir + "/results.txt", "w") as file:
        file.write(results)
    wandb.finish()

    print("Training finished")


if __name__ == '__main__':
    train()
