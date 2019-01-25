import sys
import os
import ctypes
import struct
import cv2
import glob
import time
import numpy as np
import threading
from ctypes import CDLL, c_int, c_void_p, POINTER, c_ubyte, c_ulonglong, pointer, c_char_p, create_string_buffer, addressof, CFUNCTYPE, Structure, cast, c_longlong

NORM_HEIGHT = NORM_WIDTH = 224
NUM_CHANNEL = 3
IMAGE_SCALAR = 256.0
GT_LABELS = [0, 1, 2, 3, ]

RESULT_FN = "recog.acc"

# Create Callback and Callback Args
class TimeInfo(Structure):
    _fields_ = [
            ("duration", c_int),
            ("count", c_int),
            ("correct", c_int),
            ("bgnT", c_longlong),
            ("endT", c_longlong),
            ("gt_label", c_int),
    ]

LPResultCB = CFUNCTYPE(c_int, c_void_p, POINTER(c_ubyte), c_int, c_ulonglong, c_int)

def callback_func(pPrivateData, retBuf, bufLen, pid, classNum):

    py_pPrivateData = cast(pPrivateData, POINTER(TimeInfo))
    py_retBuf = ctypes.string_at(retBuf, bufLen)

    """
    pixels = struct.unpack('%dB' % (NORM_HEIGHT*NORM_WIDTH*NUM_CHANNEL, ), py_retBuf[:NORM_HEIGHT*NORM_WIDTH*NUM_CHANNEL])
    image = np.array(pixels, dtype = np.uint8).reshape((NORM_HEIGHT, NORM_WIDTH, NUM_CHANNEL))
    #cv2.imshow("Image.Callback", image)
    """

    ts = int(time.time()*1000)
    if py_pPrivateData.contents.count == 0:
        py_pPrivateData.contents.bgnT = ts
        py_pPrivateData.contents.endT = ts

    duration = ts - py_pPrivateData.contents.endT
    py_pPrivateData.contents.endT = ts

    py_pPrivateData.contents.duration += duration
    probs = struct.unpack('4f', py_retBuf[NORM_HEIGHT*NORM_WIDTH*NUM_CHANNEL:])
    pred = np.argmax(probs)
    is_correct = 1 if pred == py_pPrivateData.contents.gt_label else 0
    py_pPrivateData.contents.count += 1
    py_pPrivateData.contents.correct += is_correct

    return 0


def get_callback():
    callback = LPResultCB(callback_func)
    callback_args = TimeInfo()
    callback_args.duration = 0
    callback_args.count = 0
    callback_args.correct = 0
    callback_args.bgnT = 0
    callback_args.endT = 0
    callback_args.gt_label = -1
    return callback, callback_args


def get_config(callback, callback_args):
    modelPath = os.path.join(os.getcwd(), 'model')
    stickCNNName = os.path.join(modelPath, "conv.dat")
    hostNetName = os.path.join(modelPath, "fc.dat")
    #print(modelPath)
    #print(stickCNNName, hostNetName)

    config = NXEngineConf()
    config.stickNum = 1
    config.threadNum = 6
    config.netType = 2
    config.classNum = 4
    config.sg_beginID = 0
    config.delayTime = 7000
    config.modelPath = bytes(modelPath, encoding = 'utf-8')
    config.stickCNNName = bytes(stickCNNName, encoding = 'utf-8')
    config.hostNetName = bytes(hostNetName, encoding = 'utf-8')
    config.pCB = callback
    config.pCBData = cast(pointer(callback_args), c_void_p)     # c_void_p(None)
    return config


def read_image(filename):
    raw_image = cv2.imread(filename)
    #print(raw_image.shape, raw_image.dtype)
    #cv2.imshow("Image.Raw", raw_image)

    mid = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    #cv2.imshow("Image.Color", mid)

    rows, cols = raw_image.shape[:2]
    scale = min(cols, rows) / IMAGE_SCALAR
    newHeight = int(rows / scale)
    newWidth = int(cols / scale)
    resized_image = cv2.resize(mid, (newWidth, newHeight))
    #cv2.imshow("Image.Resize", resized_image)

    x = int((newWidth - NORM_WIDTH) / 2.0)
    y = int((newHeight - NORM_HEIGHT) / 2.0)
    img224 = resized_image[y:(y+NORM_HEIGHT), x:(x+NORM_WIDTH), :]
    #cv2.imshow("Image.Final", img224)
    #print(img224[:10, 0, 0], type(img224), img224.shape, img224.dtype)

    #cv2.waitKey(0)
    return img224


class NXEngineConf(Structure):
    _fields_ = [
            ("stickNum", c_int),
            ("threadNum", c_int),
            ("netType", c_int),
            ("classNum", c_int),
            ("sg_beginID", c_int),
            ("delayTime", c_int),
            ("modelPath", c_char_p),
            ("stickCNNName", c_char_p),
            ("hostNetName", c_char_p),
            ("pCB", LPResultCB),
            ("pCBData", c_void_p)
    ]

# Push Task
class NXImg(Structure):
    _fields_ = [
            ("w", c_int),
            ("h", c_int),
            ("pixfmt", c_int),
            ("data", POINTER(c_ubyte))
    ]


def get_fake_samples(gt_label):
    img224 = {}
    tgt_filename = None
    for filename in glob.glob("model/val/%d/*" % gt_label):
        if tgt_filename is None:
            tgt_filename = filename
        img224[filename] = read_image(tgt_filename)
        #img224.append(read_image(filename))
    return img224

def get_samples(gt_label):
    img224 = {}
    for filename in glob.glob("model/val/%d/*" % gt_label):
        img224[filename] = read_image(filename)
        #img224.append(read_image(filename))
        #break
    return img224


def run():
    # Open DLL file
    filename = "./TEEClassifier.dll"
    hdll = CDLL(filename)

    # Param for Creation: Pointer to Engine
    p_engine = c_void_p(None)
    pp_engine = pointer(p_engine)

    # Param for Creation: Pointer to Engine Config
    callback, callback_args = get_callback()
    config = get_config(callback, callback_args)
    p_config = pointer(config)

    # Create Engine
    hdll.NXCreateInferenceEngine(pp_engine, p_config)

    pid = c_ulonglong(1)
    # Evaluate Samples for Each Class
    fp = open(RESULT_FN, 'w')
    for gt_label in GT_LABELS:
        callback_args.gt_label = gt_label
        callback_args.count = callback_args.correct = 0
        callback_args.duration = 0
        img224 = get_samples(gt_label)
        for filename, npy_img in img224.items():
            img = NXImg()
            img.w = NORM_WIDTH
            img.h = NORM_HEIGHT
            img.pixfmt = 1
            num_bytes = NORM_WIDTH * NORM_HEIGHT * NUM_CHANNEL
            #img.data = cast(npy_img.tostring(), POINTER(c_ubyte))
            c_npy_img = np.ascontiguousarray(npy_img)
            img.data = cast(c_npy_img.ctypes.data, POINTER(c_ubyte))
            # Push Task
            #pid = c_ulonglong(0)
            hdll.NXPushTask(p_engine, pointer(img), pointer(pid))
            #hdll.NXClearAllTask(p_engine)
        hdll.NXClearAllTask(p_engine)

        correct_count, total_count = callback_args.correct, callback_args.count
        acc = 100.0*correct_count/total_count if total_count > 0 else 0.0
        total_time = callback_args.duration
        mean_duration = 1.0*total_time / total_count
        fp.write("[Acc for Label %d] correct: %d, total: %d, acc: %f; [Average Time Cost] %f (ms)\n" % (gt_label, correct_count, total_count, acc, mean_duration))
    fp.close()

    # Clear
    #hdll.NXClearAllTask(p_engine)
    hdll.NXDestroyInferenceEngine(p_engine)


if __name__ == '__main__':
    run()





