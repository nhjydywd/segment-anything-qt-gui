import os
import numpy as np
import cv2
import re
import time
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
            
def replacePostfix(path, postfix, need_dot=True):
    path = path.split(".")[:-1]
    postfix = "." + postfix if need_dot else postfix
    path = "".join(path) + postfix
    return path

def removeMatchedFiles(dir, pattern):
    for file in os.listdir(dir):
        if re.match(pattern, file):
            os.remove(os.path.join(dir, file))

def ensureDir(path):
    os.makedirs(path, exist_ok=True)
    return path;

def imResizeKeepRatio(image, maxWidth, maxHeight, inter=cv2.INTER_AREA):

    height, width = image.shape[:2]
    ratioWidth = maxWidth / width
    ratioHeight = maxHeight / height
    if ratioWidth < ratioHeight:
        finalWidth = maxWidth
        finalHeight = int(height * ratioWidth)
    else:
        finalWidth = int(width * ratioHeight)
        finalHeight = maxHeight
    return cv2.resize(image, (finalWidth, finalHeight), inter)

def imResizeUndeformed(image, width, height, inter=cv2.INTER_AREA):
    image = imResizeKeepRatio(image, width, height, inter)

    shape = [*image.shape]
    shape[1] = width
    shape[0] = height

    res = np.zeros(shape, dtype=np.uint8)


    l_pad = (width - image.shape[1]) // 2
    t_pad = (height - image.shape[0]) // 2

    res[t_pad:t_pad+image.shape[0], l_pad:l_pad+image.shape[1]] = image
    return res

def imRead(path, flag):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
def imWrite(path, image):
    cv2.imencode(".png", image)[1].tofile(path)

def drawPrompts(image, points, point_labels, box):

    if not np.array_equal(points, None):
        for i in range(points.shape[0]):
            point = (int(points[i][0]), int(points[i][1]))
            label = point_labels[i]
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            
            cv2.circle(image, point, 3, color, 5)
    if not np.array_equal(box, None):
        left, top, right, bottom = box
        pt1 = (int(left), int(top))
        pt2 = (int(right), int(bottom))
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

def ensurePointsInBox(points, point_labels, box):
    if np.array_equal(box, None):
        return
    lMost = min(box[0], box[2])
    tMost = min(box[1], box[3])
    rMost = max(box[0], box[2])
    bMost = max(box[1], box[3])
    for i in range(points.shape[0]):
        if point_labels[i] == 0:
            continue
        x, y = points[i]
        lMost = min(lMost, x)
        tMost = min(tMost, y)
        rMost = max(rMost, x)
        bMost = max(bMost, y)
    box[:] = np.array([lMost, tMost, rMost, bMost])
    # return np.array([lMost, tMost, rMost, bMost])

def traverseChildren(QParent, fn):
    for child in QParent.children():
        fn(child)
        traverseChildren(child, fn)

class Timer:
    def __init__(self):
        self.start = 0
        self.end = 0
        self.interval = 0
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start



def initModelByList(model:QStandardItemModel, ls):
    model.clear()
    for (i, x) in enumerate(ls):
        item = QStandardItem(f"{i+1}")
        item.setData(x, role=Qt.ItemDataRole.UserRole)
        model.appendRow(item)

def initListByModel(model:QStandardItemModel):
    ls = []
    for i in range(model.rowCount()):
        ls.append(model.item(i).data(role=Qt.ItemDataRole.UserRole))
    return ls
    # obj.clear()
    # for i in range(target.rowCount()):
    #     target.beginInsertColumns
    #     obj.insertRow(target.row)

    
def mergeMasks(ls_mask, ls_invert):
    assert len(ls_mask) > 0 and len(ls_mask) == len(ls_invert)
    assert len(ls_mask[0].shape) == 2
    mask_positive = np.zeros(ls_mask[0].shape, dtype=np.uint8)
    mask_negative = np.zeros(ls_mask[0].shape, dtype=np.uint8)
    for mask, invert in zip(ls_mask, ls_invert):
        if invert:
            mask_negative += mask
        else:
            mask_positive += mask
    return np.logical_and(mask_positive>0, mask_negative==0)

