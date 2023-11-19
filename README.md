# pytorch-sample
ι208研究室の `(0, 0)` と `(13, 12)` の2地点をCNNで分類する、基礎的なディープラーニングモデルです。
サンプルは Python 3.10 と PyTorch 2.1.1 によって構築されています。

![PXL_20230613_071427827](https://github.com/sakusaku3939/pytorch-sample/assets/53967490/c1687d33-d2ba-4a97-99c2-03e8bc9d924b)
![PXL_20230613_071338135](https://github.com/sakusaku3939/pytorch-sample/assets/53967490/745ee9d2-145e-4ce5-ad8b-64c1c62c34ec)

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
