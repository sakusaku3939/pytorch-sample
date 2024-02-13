# pytorch-sample
ι208研究室の`(0, 0)`と`(13, 12)`の2地点をCNNで分類する深層学習モデルのサンプルです。
本リポジトリは、Python 3.10とPyTorch 2.1.1を使用します。

![PXL_20230613_071427827](https://github.com/sakusaku3939/pytorch-sample/assets/53967490/c1687d33-d2ba-4a97-99c2-03e8bc9d924b)
![PXL_20230613_071338135](https://github.com/sakusaku3939/pytorch-sample/assets/53967490/745ee9d2-145e-4ce5-ad8b-64c1c62c34ec)

## 環境構築
1. GitHubからクローンして、ディレクトリに移動します。
```
git clone https://github.com/sakusaku3939/pytorch-sample.git
cd pytorch-sample
```
<br>

2. Pythonのvenvを使用して作業ディレクトリに仮想環境を作成した後、移動します。
- Mac, Linux
```
python -m venv venv
source venv/bin/activate
```
- Windows
```
python -m venv venv
.\venv\Scripts\activate
```
<br>

3. 仮想環境の中で、必要なライブラリをインストールします。
```
pip install -r requirements.txt
```
<br>

4. Weights & Biases にログインします。
> 学習経過のグラフを見ることができるサイトです。  
> 事前に、https://www.wandb.jp/ からW&Bアカウントを作成しておいてください。
```
wandb login
```
<br>

5. 以上で、モデルの学習を開始できます！
```
python main.py
```
<br>

推論時には、以下のように main.py の`predict`のコメントアウトを外し、model_path の`20231119175655`の部分を使用したいモデルのパスに置き換えることで、テストデータへの予測精度を確認できます。
```
# train()
predict(model_path="outputs/20231119175655/model.pth")
```
