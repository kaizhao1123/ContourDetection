import os

import cv2

from Detection_DexiNed import UsingDexiNed
from Detection_HED import UsingHED
from Detection_Threshold import UsingThreshold
from Threshold_UI import CreateThresholdUI

if __name__ == '__main__':

    imageName = '0003'
    image = cv2.imread('pic_in/' + imageName + '.bmp')
    outputFolder = '3-23-23'
    outputPath = './pic_out/' + outputFolder + '/'

    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    # using HED
    # Canny_vint = 60
    # UsingHED(outputFolder, image, imageName, Canny_vint)

    # using DexiNed
    # UsingDexiNed(outputPath)

    # using Threshold
    # vint = 70
    # UsingThreshold(outputPath, image, imageName, vint)
    CreateThresholdUI(outputPath)


