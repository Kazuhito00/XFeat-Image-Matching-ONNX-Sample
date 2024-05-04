#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import cv2

from xfeat_onnx import XFeatONNX


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--xfeat_model',
        type=str,
        default='onnx_model/xfeat_640x352.onnx',
    )
    parser.add_argument(
        '--interpolator_bilinear',
        type=str,
        default='onnx_model/interpolator_bilinear_640x352.onnx',
    )
    parser.add_argument(
        '--interpolator_bicubic',
        type=str,
        default='onnx_model/interpolator_bicubic_640x352.onnx',
    )
    parser.add_argument(
        '--interpolator_nearest',
        type=str,
        default='onnx_model/interpolator_nearest_640x352.onnx',
    )

    parser.add_argument(
        '--image1',
        type=str,
        default='image/sample1.jpg',
    )
    parser.add_argument(
        '--image2',
        type=str,
        default='image/sample2.jpg',
    )

    parser.add_argument('--use_gpu', action='store_true')

    args: argparse.Namespace = parser.parse_args()

    return args


def main():
    args = get_args()

    xfeat_path = args.xfeat_model
    interp_bilinear_path = args.interpolator_bilinear
    interp_bicubic_path = args.interpolator_bicubic
    interp_nearest_path = args.interpolator_nearest

    image1_path = args.image1
    image2_path = args.image2

    use_gpu = args.use_gpu

    # XFeatモデル準備
    xfeat = XFeatONNX(
        xfeat_path,
        interp_bilinear_path,
        interp_bicubic_path,
        interp_nearest_path,
        use_gpu,
    )

    # 画像読み込み
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 特徴点抽出
    mkpts0, mkpts1 = xfeat.match(image1, image2, top_k=4096)
    result = xfeat.calc_warp_corners_and_matches(
        mkpts0,
        mkpts1,
        image1,
    )

    # デバッグ描画
    warped_corners = result[0]
    keypoints1 = result[1][0]
    keypoints2 = result[1][1]
    matches = result[1][2]

    # 画像2に変換された角を描画
    image2_with_corners = image2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(image2_with_corners, start_point, end_point, (0, 255, 0), 4)

    # 対応点を描画
    debug_image = cv2.drawMatches(
        image1,
        keypoints1,
        image2_with_corners,
        keypoints2,
        matches,
        None,
        matchColor=(0, 255, 0),
        flags=2,
    )

    cv2.imshow('', debug_image)
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()
