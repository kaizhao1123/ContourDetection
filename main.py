import os
import cv2
from Detection_DexiNed import UsingDexiNed
from Detection_HED import UsingHED
from Detection_Sobel import UsingSobel_openCV, Using2DFilter, UsingSobel_custom, drawContour_Sobel
from Detection_Threshold import UsingThreshold
from Threshold_UI import CreateThresholdUI


if __name__ == '__main__':

    imageName = '0034'
    image = cv2.imread('pic_in/' + imageName + '.bmp')
    outputFolder = '4-4-23'
    outputPath = './pic_out/' + outputFolder + '/'

    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    # ### using HED ********
    # Canny_vint = 60
    # UsingHED(outputFolder, image, imageName, Canny_vint)

    # ### using DexiNed **********
    # UsingDexiNed(outputPath)

    # ### using Threshold   *********
    # vint = 70
    # UsingThreshold(outputPath, image, imageName, vint)
    # CreateThresholdUI(outputPath)

    # ### using sobel *********
    # UsingSobel_openCV(outputFolder, image, imageName)
    # Using2DFilter(outputFolder, image, imageName)
    UsingSobel_custom(image, imageName, outputFolder)

    image = cv2.imread('pic_out/4-4-23/' + imageName + '_sobel_cus.jpg')
    # print(image)
    drawContour_Sobel(image, imageName, outputPath, vint=15)    # white dark 45, light 70; black dark 35, light 35.


