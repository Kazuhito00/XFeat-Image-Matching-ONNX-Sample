> **Note**
> <br>このリポジトリは試験的なONNX変換とサンプルの実装です
> <br>・このリポジトリで公開しているXFeatモデルはGPUでONNX推論してもあまり高速化が期待できません
> <br>・InterpolatorモデルはONNX変換していますが、Numpyなどで処理したほうが効率が良い可能性があります
> <br>・サンプルスクリプトはスパースな特徴点の抽出のみ実装しており、密な特徴点の抽出の実装はしていません

# XFeat-Image-Matching-ONNX-Sample
[XFeat](https://github.com/verlab/accelerated_features/tree/main)をONNXに変換し特徴点マッチングを行うサンプルです。<br>
ONNXに変換したモデルも同梱しています。変換自体を試したい方はColaboratoryで[XFeat-Convert2ONNX.ipynb](https://github.com/Kazuhito00/XFeat-Image-Matching-ONNX-Sample/blob/main/XFeat-Convert2ONNX.ipynb)を使用ください。

![Demo](https://github.com/Kazuhito00/XFeat-Image-Matching-ONNX-Sample/assets/37477845/44d772e5-734d-4790-b27e-85a13242b7fa)

# Requirement
* OpenCV 4.5.3.56 or later
* onnxruntime 1.13.0 or later
* scipy 1.11.4 or later

# Demo
デモの実行方法は以下です。
```bash
python sample.py
```
* --xfeat_model<br>
ロードするモデルの格納パス<br>
デフォルト：onnx_model/xfeat_640x352.onnx
* --interpolator_bilinear<br>
ロードするモデルの格納パス<br>
デフォルト：onnx_model/interpolator_bilinear_640x352.onnx
* --interpolator_bicubic<br>
ロードするモデルの格納パス<br>
デフォルト：onnx_model/interpolator_bicubic_640x352.onnx
* --interpolator_nearest<br>
ロードするモデルの格納パス<br>
デフォルト：onnx_model/interpolator_nearest_640x352.onnx
* --image1<br>
特徴点抽出する画像の格納パス<br>
デフォルト：image/sample1.jpg
* --image2<br>
特徴点抽出する画像の格納パス<br>
デフォルト：image/sample1.jpg

# Reference
* [verlab/accelerated_features](https://github.com/verlab/accelerated_features)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)

# License
XFeat-Image-Matching-ONNX-Sample is under [Apache-2.0 license](LICENSE).
