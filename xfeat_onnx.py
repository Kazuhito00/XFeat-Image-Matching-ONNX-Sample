#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
import onnxruntime  # type: ignore
from scipy.ndimage import maximum_filter  # type: ignore


class XFeatONNX(object):

    def __init__(
        self,
        xfeat_path: str,
        interp_bilinear_path: str,
        interp_bicubic_path: str,
        interp_nearest_path: str,
        use_gpu: bool,
    ):
        # GPU使用の設定。GPUが利用可能な場合は、CUDAプロバイダーを含める。
        providers: List[str] = ['CPUExecutionProvider']
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # ONNXモデルのロード
        self.xfeat = onnxruntime.InferenceSession(
            xfeat_path,
            providers=providers,
        )

        self.interp_bilinear = onnxruntime.InferenceSession(
            interp_bilinear_path,
            providers=providers,
        )
        self.interp_bicubic = onnxruntime.InferenceSession(
            interp_bicubic_path,
            providers=providers,
        )
        self.interp_nearest = onnxruntime.InferenceSession(
            interp_nearest_path,
            providers=providers,
        )

        # 入力画像の幅と高さを取得
        self.input_width: int = self.xfeat.get_inputs()[0].shape[3]
        self.input_height: int = self.xfeat.get_inputs()[0].shape[2]

        # インターポレーター用の入力名
        self.interp_input_name1: str = self.interp_nearest.get_inputs()[0].name
        self.interp_input_name2: str = self.interp_nearest.get_inputs()[1].name

    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        top_k: int = 4096,
        min_cossim: float = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 特徴点と記述子を抽出
        result1 = self._detect_and_compute(
            self.xfeat,
            image1,
            top_k=top_k,
        )[0]
        result2 = self._detect_and_compute(
            self.xfeat,
            image2,
            top_k=top_k,
        )[0]

        # 特徴点のマッチング
        indexes1, indexes2 = self._match_mkpts(
            result1['descriptors'],
            result2['descriptors'],
            min_cossim=min_cossim,
        )

        # マッチング結果の特徴点座標
        mkpts0 = result1['keypoints'][indexes1]
        mkpts1 = result2['keypoints'][indexes2]

        return mkpts0, mkpts1

    def _preprocess_image(self, image: Any) -> Tuple[Any, float, float]:
        image_width: int = image.shape[1]
        image_height: int = image.shape[0]

        # 画像のリサイズと正規化
        input_image = cv2.resize(image, (self.input_width, self.input_height))
        input_image = input_image.astype(np.float32)
        input_image /= 255.0
        input_image = input_image[None, ...]
        input_image = np.transpose(input_image, (0, 3, 1, 2))

        resize_rate_w: float = image_width / self.input_width
        resize_rate_h: float = image_height / self.input_height

        return input_image, resize_rate_w, resize_rate_h

    def _get_kpts_heatmap(
        self,
        kpts: np.ndarray,
        softmax_temp: float = 1.0,
    ) -> np.ndarray:
        # キーポイントヒートマップの生成
        kpts = np.exp(kpts * softmax_temp)
        scores = kpts / np.sum(kpts, axis=1, keepdims=True)
        scores = scores[:, :64]
        B, _, H, W = scores.shape
        heatmap = np.transpose(scores, (0, 2, 3, 1)).reshape(B, H, W, 8, 8)
        heatmap = np.transpose(heatmap,
                               (0, 1, 3, 2, 4)).reshape(B, 1, H * 8, W * 8)
        return heatmap

    def _nms(
        self,
        x: np.ndarray,
        threshold: float = 0.05,
        kernel_size: int = 5,
    ) -> np.ndarray:
        # Non-Maximum Suppressionを行いキーポイントを抽出
        B, _, H, W = x.shape
        local_max = maximum_filter(
            x,
            size=(1, 1, kernel_size, kernel_size),
            mode='constant',
        )
        pos = (x == local_max) & (x > threshold)

        # キーポイント位置のバッチ処理
        pos_batched = [np.fliplr(np.argwhere(k)[:, 1:]) for k in pos]

        pad_val = max(len(k) for k in pos_batched)
        pos_array = np.zeros((B, pad_val, 2), dtype=int)

        for b, kpts in enumerate(pos_batched):
            pos_array[b, :len(kpts)] = kpts

        return pos_array

    def _detect_and_compute(
        self,
        net: onnxruntime.InferenceSession,
        x: np.ndarray,
        top_k: int = 4096,
    ) -> List[Dict[str, Any]]:
        # 画像の前処理
        x, resize_rate_w, resize_rate_h = self._preprocess_image(x)

        # バッチサイズと画像の次元を取得
        B, _, _, _ = x.shape

        # ネットワークの入力名を取得
        input_name = net.get_inputs()[0].name
        # ONNXモデルを実行し、特徴マップとキーポイントのログを取得
        M1, K1, _ = net.run(None, {input_name: x})

        # 特徴マップのL2正規化
        norm = np.linalg.norm(M1, axis=1, keepdims=True)
        M1 = M1 / norm

        # キーポイントのヒートマップを取得
        K1h = self._get_kpts_heatmap(K1)
        # 非最大抑制を適用し、キーポイントを抽出
        mkpts = self._nms(K1h, threshold=0.05, kernel_size=5)

        # インターポレーションを用いて信頼スコアを計算
        nearest_result = self.interp_nearest.run(
            None, {
                self.interp_input_name1: K1h.astype(np.float32),
                self.interp_input_name2: mkpts.astype(np.float32)
            })[0]
        bilinear_result = self.interp_bilinear.run(
            None, {
                self.interp_input_name1: K1h.astype(np.float32),
                self.interp_input_name2: mkpts.astype(np.float32)
            })[0]
        scores = (nearest_result * bilinear_result).squeeze(-1)
        # 無効なキーポイントはスコアを-1に設定
        scores[np.all(mkpts == 0, axis=-1)] = -1

        # スコアに基づいてトップKキーポイントを選択
        idxs = np.argsort(-scores)
        mkpts_x = np.take_along_axis(mkpts[..., 0], idxs, axis=-1)[:, :top_k]
        mkpts_y = np.take_along_axis(mkpts[..., 1], idxs, axis=-1)[:, :top_k]
        mkpts = np.stack([mkpts_x, mkpts_y], axis=-1)
        scores = np.take_along_axis(scores, idxs, axis=-1)[:, :top_k]

        # キーポイント位置で特徴量を補間
        feats = self.interp_bicubic.run(
            None, {
                self.interp_input_name1: M1.astype(np.float32),
                self.interp_input_name2: mkpts.astype(np.float32)
            })[0]

        # 特徴量のL2正規化
        norm = np.linalg.norm(feats, axis=-1, keepdims=True)
        feats = feats / norm

        # キーポイント座標のスケーリング調整
        mkpts = mkpts.astype(np.float32)
        mkpts *= np.array([resize_rate_w, resize_rate_h])[None, None, :]

        # 有効なキーポイントのみを抽出
        valid = scores > 0
        result = []
        for b in range(B):
            valid_b = valid[b]
            result.append({
                'keypoints': mkpts[b][valid_b],
                'scores': scores[b][valid_b],
                'descriptors': feats[b][valid_b]
            })

        return result

    def _match_mkpts(
        self,
        feats1: np.ndarray,
        feats2: np.ndarray,
        min_cossim: float = 0.82,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 特徴量間のコサイン類似度を計算
        cossim = feats1 @ feats2.T
        cossim_t = feats2 @ feats1.T

        # 相互に最大の類似度を持つインデックスを取得
        match12 = np.argmax(cossim, axis=1)
        match21 = np.argmax(cossim_t, axis=1)

        # 相互マッチングを確認
        idx0 = np.arange(len(match12))
        mutual = (match21[match12] == idx0)

        # 最小類似度以上のマッチのみを抽出
        if min_cossim > 0:
            max_cossim = np.max(cossim, axis=1)
            good = (max_cossim > min_cossim)
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1

    def calc_warp_corners_and_matches(
        self,
        ref_points: np.ndarray,
        dst_points: np.ndarray,
        image1: np.ndarray,
    ) -> Tuple[np.ndarray, List[Any]]:
        # ホモグラフィ行列を計算
        H, mask = cv2.findHomography(
            ref_points,
            dst_points,
            cv2.USAC_MAGSAC,
            3.5,
            maxIters=1_000,
            confidence=0.999,
        )
        mask = mask.flatten()

        # 画像1の角を取得
        h, w = image1.shape[:2]
        corners_image1 = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=np.float32).reshape(-1, 1, 2)

        # 画像2の座標空間に画像1の角を変換
        warped_corners = cv2.perspectiveTransform(corners_image1, H)

        # キーポイントとマッチングを準備
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
        keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

        return warped_corners, [keypoints1, keypoints2, matches]
