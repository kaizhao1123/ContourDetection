import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import math


fontSize_label = 0  # the font of the content
imageName = '0001'
image = cv2.imread('pic_in/' + imageName + '.bmp')


def createWindowForDeviceProperty(output_path):
    # get data of device property
    data = [0, 258, 140, 255, 255, 105, 108, 42, 101.8, 3.96, 36, 200, 200, 80]
    default_crop_start_line = int(data[0])
    default_crop_line = int(data[1])
    default_color_threshold_high = [int(data[2]), int(data[3]), int(data[4])]
    default_color_threshold_low = [int(data[5]), int(data[6]), int(data[7])]

    # ######### create window of device property ################
    window = tk.Tk()
    windowHeight = int(window.winfo_screenheight() / 1.3)
    windowWidth = int(window.winfo_screenwidth() / 2)
    window.title('Set up device property')
    window.geometry('%sx%s+10+10' % (windowWidth, windowHeight))
    window.resizable(width=False, height=False)

    global fontSize_label
    fontSize_label = int(window.winfo_screenwidth() / 100)  # the font of the labels
    windowHeight = int(window.winfo_screenheight() / 1.3)
    windowWidth = int(window.winfo_screenwidth() / 2)
    window_propertyHeight = int(windowHeight / 1.5)

    # locate the position of each element and the gap between them ############################
    midGap = int(windowWidth / 20)  # col gap
    eleWidth_label = int(windowWidth / 5)  # element's width
    eleHeight_label = int(window_propertyHeight / 16)  # element's height
    start_X = int(midGap/2)  # the left border
    start_Y = 10  # the top border
    secondCol_X = int(eleWidth_label/1.1)  # the second col
    thirdCol_X = int(windowWidth / 2.8)  # the third col
    fourthCol_X = int(windowWidth / 1.9)  # the fourth col
    rowGap = int(eleHeight_label * 1.4)  # the gap between row

    #  get image
    start_Y += rowGap*2
    tk.Label(window, text='Image Name: ', font=('Arial', fontSize_label)).place(x=start_X, y=start_Y)
    val_img_name = tk.StringVar()
    val_img_name.set('0001')
    entry_img_name = tk.Entry(window, textvariable=val_img_name, font=('Arial', fontSize_label))
    entry_img_name.place(x=secondCol_X, y=start_Y, width=eleWidth_label / 2.2)

    val_img_name_ex = tk.StringVar()
    val_img_name_ex.set('0')
    entry_img_name_ex = tk.Entry(window, textvariable=val_img_name_ex, font=('Arial', fontSize_label))
    entry_img_name_ex.place(x=thirdCol_X, y=start_Y, width=eleWidth_label / 2.2)

    def getImage():
        global imageName, image
        imageName = val_img_name.get()
        imageName_ex = val_img_name_ex.get()

        image = cv2.imread('pic_in/' + imageName + '.bmp')
        # save the original image
        temp = int(imageName) + int(imageName_ex)
        if temp < 100:
            imageName = '00' + str(temp)
        else:
            imageName = '0' + str(temp)

        cv2.imwrite(output_path + imageName + '.bmp', image)
        display_currentImage()
        display_contourImage()


    start_Y += rowGap
    button_GetImage = tk.Button(window, text='Get', font=('Arial', int(fontSize_label*0.9)), command=getImage)
    button_GetImage.place(x=secondCol_X, y=start_Y, width=eleWidth_label / 2.2, height=eleHeight_label)

    # ################################### display camera image #################################
    start_Y += rowGap
    frame_video = tk.Frame(window)
    frame_video.place(x=start_X, y=start_Y, width = windowWidth/2)
    label_video = tk.Label(frame_video)
    label_video.grid()

    # display one image with current camera configure.
    def display_currentImage():
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(display_image)
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)

    display_currentImage()

    # ###### Crop Image #####
    start_Y += rowGap
    start_Y += int(eleHeight_label * 10.5)
    tk.Label(window, text='Crop Line: ', font=('Arial', fontSize_label)).place(x=start_X, y=start_Y)
    val_crop_line = tk.StringVar()
    val_crop_line.set(str(default_crop_line))
    entry_crop_line = tk.Entry(window, textvariable=val_crop_line, font=('Arial', fontSize_label))
    entry_crop_line.place(x=secondCol_X, y=start_Y, width=eleWidth_label / 2.2)

    # get updated crop line
    def get_crop_line():
        try:
            val_crop = int(val_crop_line.get())
        except:
            val_crop = default_crop_line
        return val_crop

    # crop line refresh button
    def crop_line_refresh():
        cropLine = get_crop_line()
        cropImage = image.copy()
        cropImage = cropImage[default_crop_start_line:cropLine, 0:720]
        display_image = cv2.cvtColor(cropImage, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(display_image)
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)
        display_contourImage()

    button_crop_line = tk.Button(window, text='Refresh', font=('Arial', int(fontSize_label*0.9)),
                                 command=crop_line_refresh)
    button_crop_line.place(x=secondCol_X, y=start_Y, width=eleWidth_label / 2.2, height=eleHeight_label)

    # ################################### display contour image #################################
    frame_contour = tk.Frame(window)
    frame_contour.place(x=fourthCol_X, y=10)
    label_contour = tk.Label(frame_contour)
    label_contour.grid()

    # display the contour image based on the default color threshold.
    def display_contourImage():
        val_crop = get_crop_line()

        contourImage = getContour(image[default_crop_start_line:val_crop, :], imageName, default_color_threshold_high,
                                  default_color_threshold_low, output_path)
        contourImage = cv2.cvtColor(contourImage, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(contourImage)
        imgtk = ImageTk.PhotoImage(image=img)
        label_contour.imgtk = imgtk
        label_contour.configure(image=imgtk)

    display_contourImage()

    # ###### threshold high #####
    start_Y -= rowGap
    tk.Label(window, text='Thrd H: ', font=('Arial', fontSize_label)).place(x=fourthCol_X, y=start_Y)
    val_thrd_high_b = tk.StringVar()
    val_thrd_high_g = tk.StringVar()
    val_thrd_high_r = tk.StringVar()
    val_thrd_high_b.set(str(default_color_threshold_high[0]))
    val_thrd_high_g.set(str(default_color_threshold_high[1]))
    val_thrd_high_r.set(str(default_color_threshold_high[2]))
    entry_thrd_high_b = tk.Entry(window, textvariable=val_thrd_high_b, font=('Arial', fontSize_label),
                                 fg="blue")
    entry_thrd_high_b.place(x=fourthCol_X + eleWidth_label / 1.8, y=start_Y, width=eleWidth_label / 2.2)
    entry_thrd_high_g = tk.Entry(window, textvariable=val_thrd_high_g, font=('Arial', fontSize_label),
                                 fg="green")
    entry_thrd_high_g.place(x=fourthCol_X + eleWidth_label / 0.9, y=start_Y, width=eleWidth_label / 2.2)
    entry_thrd_high_r = tk.Entry(window, textvariable=val_thrd_high_r, font=('Arial', fontSize_label),
                                 fg="red")
    entry_thrd_high_r.place(x=fourthCol_X + eleWidth_label / 0.6, y=start_Y, width=eleWidth_label / 2.2)

    # ###### threshold low #####
    start_Y += rowGap
    tk.Label(window, text='Thrd L: ', font=('Arial', fontSize_label)).place(x=fourthCol_X, y=start_Y)
    val_thrd_low_b = tk.StringVar()
    val_thrd_low_g = tk.StringVar()
    val_thrd_low_r = tk.StringVar()
    val_thrd_low_b.set(str(default_color_threshold_low[0]))
    val_thrd_low_g.set(str(default_color_threshold_low[1]))
    val_thrd_low_r.set(str(default_color_threshold_low[2]))
    entry_thrd_low_b = tk.Entry(window, textvariable=val_thrd_low_b, font=('Arial', fontSize_label),
                                fg="blue")
    entry_thrd_low_b.place(x=fourthCol_X + eleWidth_label / 1.8, y=start_Y, width=eleWidth_label / 2.2)
    entry_thrd_low_g = tk.Entry(window, textvariable=val_thrd_low_g, font=('Arial', fontSize_label),
                                fg="green")
    entry_thrd_low_g.place(x=fourthCol_X + eleWidth_label / 0.9, y=start_Y, width=eleWidth_label / 2.2)
    entry_thrd_low_r = tk.Entry(window, textvariable=val_thrd_low_r, font=('Arial', fontSize_label),
                                fg="red")
    entry_thrd_low_r.place(x=fourthCol_X + eleWidth_label / 0.6, y=start_Y, width=eleWidth_label / 2.2)

    start_Y += int(fontSize_label*2)
    tk.Label(window, text='(B G R)', font=('Arial', int(fontSize_label*0.9))).place(x=fourthCol_X,
                                                                                                    y=start_Y)

    # get updated color threshold
    def get_threshold():
        data_high = []
        data_low = []

        try:
            h_b = int(val_thrd_high_b.get())
        except:
            h_b = 0
        try:
            h_g = int(val_thrd_high_g.get())
        except:
            h_g = 0
        try:
            h_r = int(val_thrd_high_r.get())
        except:
            h_r = 0
        data_high.append(h_b)
        data_high.append(h_g)
        data_high.append(h_r)

        try:
            l_b = int(val_thrd_low_b.get())
        except:
            l_b = 0
        try:
            l_g = int(val_thrd_low_g.get())
        except:
            l_g = 0
        try:
            l_r = int(val_thrd_low_r.get())
        except:
            l_r = 0
        data_low.append(l_b)
        data_low.append(l_g)
        data_low.append(l_r)

        return data_high, data_low

    # color threshold button
    def threshold_refresh():
        val_crop = get_crop_line()
        data_high, data_low = get_threshold()
        contourImage = getContour(image[default_crop_start_line:val_crop, :], imageName, data_high, data_low, output_path)
        contourImage = cv2.cvtColor(contourImage, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(contourImage)
        imgtk = ImageTk.PhotoImage(image=img)
        label_contour.imgtk = imgtk
        label_contour.configure(image=imgtk)

    button_threshold = tk.Button(window, text='Refresh', font=('Arial', int(fontSize_label*0.9)),
                                 command=threshold_refresh)
    button_threshold.place(x=fourthCol_X + eleWidth_label / 0.6, y=start_Y, width=eleWidth_label / 2.2,
                           height=eleHeight_label)

    # ########  Save button #############
    start_Y += rowGap

    def button_save_setting():
        pass

    button_save = tk.Button(window, text='Save', font=('Arial', int(fontSize_label*0.9)),
                            command=button_save_setting)
    button_save.place(x=thirdCol_X, y=start_Y, width=eleWidth_label / 2,
                      height=eleHeight_label)

    def button_cancel_setting():
        window.destroy()

    button_cancel = tk.Button(window, text='Cancel', font=('Arial', int(fontSize_label*0.9)),
                              command=button_cancel_setting)
    button_cancel.place(x=fourthCol_X, y=start_Y, width=eleWidth_label / 2, height=eleHeight_label)

    window.mainloop()


# get the contour image
def getContour(img, imageName, thrd_high, thrd_low, output_path):

    image = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(image, 0, 50)  # using low vint value
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # combine the edges image and original image
    contour_image = cv2.add(edges, img)
    cv2.imwrite(output_path + imageName + '_combinCanny.jpg', contour_image)

    # get binary image without color blue.
    hsv = cv2.cvtColor(contour_image, cv2.COLOR_BGR2HSV)
    upper_blue = np.array(thrd_high)
    lower_blue = np.array(thrd_low)
    binary_image = cv2.inRange(hsv, lower_blue, upper_blue)
    binary_image = cv2.bitwise_not(binary_image)

    # # # draw the contour
    thresh = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY)[1]
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
    contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    big_contour = max(contours, key=cv2.contourArea)
    result_image = img.copy()
    cv2.drawContours(result_image, [big_contour], 0, (0, 0, 255), thickness=1)

    h, w, _ = image.shape
    template = np.zeros([h, w, 3], dtype=np.uint8)
    cv2.drawContours(template, [big_contour], 0, (255, 255, 255), thickness=1)
    cv2.imwrite(output_path + imageName + '.jpg', template)
    print('update')
    return result_image


def CreateThresholdUI(output_path):

    createWindowForDeviceProperty(output_path)
