import os
import torch
import torch.optim as optim
import random
import numpy as np
import torchvision
import wandb
import matplotlib.pyplot
from torch import nn
from tqdm import tqdm
from datetime import datetime

from simple_cnn import SimpleCNN
from validation_functions import get_classification_accuracy
from dataset_loader import load_image, load_test_image

num_epochs = 8  # 学習の回数
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
    train_loader, valid_loader = load_image(batch_size, num_workers, random_state)

    for epoch in range(1, num_epochs + 1):
        running_loss, running_score = 0.0, 0.0
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
            for j, data in tqdm(enumerate(valid_loader, 0)):
                inputs = data[0].to(device)
                labels = data[1].to(device)

                # モデルの精度（正解率）を検証する
                pred = model(inputs)
                running_score += get_classification_accuracy(pred, labels)

        # 各エポック結果の平均を記録
        epoch_loss = running_loss / len(train_loader)
        epoch_score = running_score / len(valid_loader)
        wandb.log({"Epoch": epoch, "Loss": epoch_loss, "Score": epoch_score})
        results += ("Epoch:" + str(epoch) + "  " + f"Loss: {epoch_loss}  Score: {epoch_score}\n")

        print(f"Loss: {epoch_loss}  Score: {epoch_score}\n")

    # モデル学習完了後の保存
    torch.save(model.state_dict(), output_dir + "/model.pth")
    with open(output_dir + "/results.txt", "w") as file:
        file.write(results)
    wandb.finish()
    print(f"Output file to {output_dir}/model.pth\n")

    print("Training finished")


def predict(model_path):
    # CUDA（GPU）を使用するように切り替える
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.multiprocessing.freeze_support()

    # 分類クラスの一覧
    classes = ["(0, 0)", "(13, 12)"]
    classes_correct_n = [0, 0]
    classes_total_n = [0, 0]

    # CNNモデルを指定
    model = SimpleCNN()
    model = model.to(device)
    model = model.eval()

    # 学習済みモデルのパラメータを読み込み
    model.load_state_dict(torch.load(model_path))

    # テストデータの読み込み
    test_loader = load_test_image(batch_size, num_workers)

    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # モデルによる推論を行い、数字が大きい方のindexを取得する
            pred = model(inputs)
            _, pred = torch.max(pred.data, dim=1)

            for i in range(batch_size):
                if pred[i] == labels[i]:
                    # 分類に成功した数をカウント
                    classes_correct_n[labels[i]] += 1
                else:
                    # 分類に失敗した画像を表示する
                    show_img(torchvision.utils.make_grid(inputs[i]))
                classes_total_n[labels[i]] += 1

    # 各クラスごとの正解率を表示
    for i in range(2):
        accuracy = classes_correct_n[i] / classes_total_n[i]
        print(f"Accuracy of {classes[i]} : {100 * accuracy} %")


def show_img(img):
    img = img / 2 + 0.5
    numpy_img = img.numpy()
    matplotlib.pyplot.imshow(np.transpose(numpy_img, (1, 2, 0)))
    matplotlib.pyplot.show()


if __name__ == '__main__':
    train()
    # predict(model_path="outputs/20231119175655/model.pth")
