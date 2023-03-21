import math
import cv2
import numpy as np
from PIL import Image


def UsingThreshold(output_folder, image, imageName, vint):
    type = imageName[:5]
    print(type)
    # img1 = cv2.imread('./pic/t2/' + type + '.bmp')

    image = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(image, 0, 50)  # using low vint value
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    h = edges.shape[0]
    edges_adjust = np.zeros_like(edges)
    edges_adjust[0:h - 5, :] = edges[0:h - 5, :]

    # combine the edges image and original image
    contour_image = cv2.add(edges_adjust, image)
    # contour_image = cv2.add(edges, img1)
    cv2.imwrite('./pic_out/' + output_folder + '/' + imageName + '_Thre_contour.jpg', contour_image)

    # get binary image without color blue.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 110, 0])
    upper_blue = np.array([140, 255, 255])
    binary_image = cv2.inRange(hsv, lower_blue, upper_blue)
    masked_image = np.copy(image)
    masked_image[binary_image != 0] = [0, 0, 0]
    cv2.imwrite('./pic_out/' + output_folder + '/' + imageName + '_Thre_binary.jpg', masked_image)
    binary_image = cv2.bitwise_not(binary_image)

    #
    if type == 'black':
        binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # black background
        thresh = cv2.threshold(binary_image, vint, 255, cv2.THRESH_BINARY)[1]
    else:
        thresh = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY)[1]

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
    contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    big_contour = max(contours, key=cv2.contourArea)
    result_image_contour = image.copy()
    cv2.drawContours(result_image_contour, [big_contour], 0, (0, 0, 255), thickness=1)
    cv2.imwrite('./pic_out/' + output_folder + '/' + imageName + '_Thre_drawCon.jpg', result_image_contour)

    #
    h, w, _ = image.shape
    template = np.zeros([h, w, 3], dtype=np.uint8)
    cv2.drawContours(template, [big_contour], 0, (255, 255, 255), thickness=1)
    cv2.imwrite('./pic_out/' + output_folder + '/' + imageName + '_Thre_drawCon_binary.jpg', template)

    # 1. get the height

    # detect the size of the biggest contour using opencv.
    thresh = cv2.threshold(binary_image, 70, 255, cv2.THRESH_BINARY)[1]
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
    contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    big_contour = max(contours, key=cv2.contourArea)
    maxRect = cv2.boundingRect(big_contour)
    height = maxRect[3]
    # print(maxRect)

    # 2. get the X, Y, width
    binary_image_new = binary_image[0:binary_image.shape[0]-10, 0:720]
    thresh = cv2.threshold(binary_image_new, 70, 255, cv2.THRESH_BINARY)[1]
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
    contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    big_contour = max(contours, key=cv2.contourArea)
    maxRect = cv2.boundingRect(big_contour)
    # print(maxRect)

    X, Y, width, _ = maxRect

    # print(X, Y, width, height)

    firstCrop = image[Y: Y + height, X: X + width]  # get the target

    # binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    firstCrop_binary = binary_image[Y: Y + height, X: X + width]

    H = firstCrop.shape[0]
    W = firstCrop.shape[1]

    imageHeight = 200
    imageWidth = 200

    # create the black image with special size to carry the target.
    cropResult = np.zeros([imageHeight, imageWidth, 3], dtype=np.uint8)
    start_Y = math.ceil((imageHeight - H) / 2)
    start_X = math.ceil((imageWidth - W) / 2)
    mask_image = np.zeros([imageHeight, imageWidth], dtype=np.uint8)

    bottomY = 0
    # make the new target in the center of the result
    if bottomY == 0:
        cropResult[start_Y: start_Y + H, start_X: start_X + W] = firstCrop
        mask_image[start_Y: start_Y + H, start_X: start_X + W] = firstCrop_binary
    else:
        cropResult[bottomY - H: bottomY, start_X: start_X + W] = firstCrop
        mask_image[start_Y: start_Y + H, start_X: start_X + W] = firstCrop_binary

    # #####  black
    if type == 'black':
        Hint = [0, 255]
        Sint = [0, 255]
        Vint = [vint, 255]
        v = str(vint)+'_'
        temp = Image.fromarray(mask_image)
        mask = HSVSegment(cropResult, Hint, Sint, Vint)
        mask_image = Image.fromarray(mask)
        mask_image.save('./pic_out/' + output_folder + '/' + imageName + '_Thre_mask_' + v + '.png')
    else:
        cv2.imwrite('./pic_out/' + output_folder + '/' + imageName + '_Thre_mask.jpg', mask_image)
    # draw white filled contour on black background
    # result = cropResult.copy() # np.zeros_like(mask) cv2.bitwise_not(cropResult)

    # save results
    cv2.imwrite('./pic_out/' + output_folder + '/' + imageName + '_Thre_crop.jpg', cropResult)


def HSVSegment(rgb_image, Hint, Sint, Vint):
    # convert RGB image to HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    sz = hsv_image.shape

    # define masks for each channel and set pixels with the respective
    # channel value in the given interval to foreground, i.e. 255
    hmask = np.zeros((sz[0], sz[1]), int)
    hmask[(hsv_image[:, :, 0] >= Hint[0]) & (hsv_image[:, :, 0] <= Hint[1])] = 255
    smask = np.zeros((sz[0], sz[1]), int)
    smask[(hsv_image[:, :, 1] >= Sint[0]) & (hsv_image[:, :, 1] <= Sint[1])] = 255
    vmask = np.zeros((sz[0], sz[1]), int)
    vmask[(hsv_image[:, :, 2] >= Vint[0]) & (hsv_image[:, :, 2] <= Vint[1])] = 255
    # combine channel masks to a single mask
    return hmask * smask * vmask
