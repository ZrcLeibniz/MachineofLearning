import trainLicenseProvince as A
import trainLicenseLetters as B
import trainLicenseDigits as C
import numpy as np
import tensorflow as tf

from PIL import Image



if __name__ == '__main__':
    license_num = ""
    for n in range(1, 8):
        path = "tf_car_license_dataset/test_images/2%s.bmp" % (n)
        if n == 1:
            license_num += A.predictImg(path)
        elif n == 2:
            license_num += B.predictImg(path)
        else:
            license_num += C.predictImg(path)
    print("计算机视觉,识别车牌号为: 【%s】" % license_num)
