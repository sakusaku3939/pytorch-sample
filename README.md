# pytorch-sample
ι208研究室の `(0, 0)` と `(13, 12)` の2地点をCNNで分類する、基礎的なディープラーニングモデルです。
サンプルは Python 3.10 と PyTorch 2.1.1 によって構築されています。

![PXL_20230613_071427827](https://github.com/sakusaku3939/pytorch-sample/assets/53967490/91b00adc-6a81-4b76-8268-b94d0dc28226)
![PXL_20230613_071338135](https://github.com/sakusaku3939/pytorch-sample/assets/53967490/99dcfe10-c5d2-4766-a782-6f4185e9c733)

## 環境構築
1. GitHubからクローンして、ディレクトリに移動します。
```
git clone https://github.com/sakusaku3939/pytorch-sample.git
cd pytorch-sample
```
<br>

2. Pythonのvenvを使用して、作業ディレクトリに仮想環境を作成します。
```
python -m venv venv
```
<br>

3. 仮想環境に移動後、必要なライブラリをインストールします。
```
.\venv\Scripts\activate
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
