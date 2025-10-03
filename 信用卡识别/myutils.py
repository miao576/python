# myutils.py
import cv2

def sort_contours(cnts, method="left-to-right"):
    """
    按指定方向排序轮廓
    :param cnts: 轮廓列表
    :param method: 排序方向字符串
    :return: (排序后的轮廓, 对应 boundingBox)
    """
    reverse = False
    i = 0  # 按 x 排序

    if method in ("right-to-left", "bottom-to-top"):
        reverse = True
    if method in ("top-to-bottom", "bottom-to-top"):
        i = 1  # 按 y 排序

    # 计算每个轮廓的 boundingBox
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # 同时排序
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i],
                                        reverse=reverse))
    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    等比例缩放图像
    """
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)
