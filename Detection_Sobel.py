import cv2
import numpy as np
from PIL import Image


def UsingSobel_openCV(output_folder, img, imageName):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    # Sobel Edge Detection
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    cv2.imwrite('pic_out/' + output_folder + '/' + imageName + '_Sobel_openCV.jpg', sobelxy)


def Using2DFilter(output_folder, img, imageName):

    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    median_3 = cv2.medianBlur(img, 5)

    # kernel = np.ones((3, 3), np.float32)
    kernel = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]])

    # apply on the input image, here grayscale input
    # img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    dst = cv2.filter2D(img_blur, -1, kernel)
    cv2.imwrite('pic_out/' + output_folder + '/' + imageName + '_2d_filter_out.jpg', dst)

    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    cv2.imwrite('pic_out/' + output_folder + '/' + imageName + '_Sobel_out_opencv.jpg', sobelxy)

    edges_ori = cv2.Canny(img, 0, 60)
    edges_ori = cv2.cvtColor(edges_ori, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('pic_out/' + output_folder + '/' + imageName + '_Canny_out_ori.jpg', edges_ori)

    edges = cv2.Canny(img_blur, 0, 60)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('pic_out/' + output_folder + '/' + imageName + '_Canny_out_blur.jpg', edges)

    log = cv2.Laplacian(dst, cv2.CV_64F, ksize=5)
    cv2.imwrite('pic_out/' + output_folder + '/' + imageName + '_lap.jpg', log)

    cv2.imwrite('pic_out/' + output_folder + '/' + imageName + '_median.jpg', median_3)


#
def UsingSobel_custom(image, imageName, output_folder):

    rows, columns, _ = np.shape(image)
    print(rows, columns)
    # sharpen the image
    kernel = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]])/4
    image = cv2.filter2D(image, -1, kernel)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    img = image
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b_img, g_img, r_img = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    b_img = sobel_filter(rows, columns, b_img, 0)
    g_img = sobel_filter(rows, columns, g_img, 1)
    r_img = sobel_filter(rows, columns, r_img, 2)

    temp = cv2.add(b_img, g_img)
    temp = cv2.add(r_img, temp)
    cv2.imwrite('pic_out/' + output_folder + '/' + imageName + '_sobel_cus.jpg', temp)


# sobel filter
def sobel_filter(rows, columns, image, channel):
    # adjustment(increase 1 on row) for the edge of the image(not the seed)
    rows += 1
    sobel_filtered_image = np.zeros(shape=(rows, columns))
    temp = sobel_filtered_image.copy()
    temp[:rows - 1, :] = image
    temp[rows - 1:rows, :] = image[rows - 2:rows - 1, :]

    # Here we define the matrices associated with the "Sobel filter"
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    # Now we "sweep" the image in both x and y directions and compute the output
    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, temp[i:i + 3, j:j + 3]))  # x direction
            gy = np.sum(np.multiply(Gy, temp[i:i + 3, j:j + 3]))  # y direction
            # sobel_filtered_image[i + 1, j + 1] = round(np.sqrt(gx ** 2 + gy ** 2))  # calculate the "hypotenuse"
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
    sobel_filtered_image = sobel_filtered_image[:rows - 1, :]

    res = np.zeros(shape=(rows - 1, columns, 3))
    res[:, :, channel] = sobel_filtered_image
    return res


#
def drawContour_Sobel(img, imageName, output_path, vint):
    image = cv2.GaussianBlur(img, (3, 3), 0)

    # get binary image without color blue.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # # draw the contour
    thresh = cv2.threshold(gray, vint, 255, cv2.THRESH_BINARY)[1]
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
    contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    big_contour = max(contours, key=cv2.contourArea)
    result_image = img.copy()
    cv2.drawContours(result_image, [big_contour], 0, (0, 0, 255), thickness=1)

    h, w, _ = image.shape
    template = np.zeros([h, w, 3], dtype=np.uint8)
    cv2.drawContours(template, [big_contour], 0, (255, 255, 255), thickness=1)
    cv2.imwrite(output_path + imageName + '_sobelContour.jpg', template)

    image1 = cv2.imread('pic_out/4-3-23/' + imageName + '.bmp')
    cv2.drawContours(image1, [big_contour], 0, (255, 255, 255), thickness=1)
    cv2.imwrite(output_path + imageName + '_sobel_ori.jpg', image1)

    return result_image

