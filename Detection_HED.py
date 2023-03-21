import cv2
import numpy as np
from time import time


def UsingHED(output_folder, image, imageName, CannyVint):

    class CropLayer(object):
        def __init__(self, params, blobs):
            self.xstart = 0
            self.xend = 0
            self.ystart = 0
            self.yend = 0

        # Our layer receives two inputs. We need to crop the first input blob
        # to match a shape of the second one (keeping batch size and number of channels)
        def getMemoryShapes(self, inputs):
            inputShape, targetShape = inputs[0], inputs[1]
            batchSize, numChannels = inputShape[0], inputShape[1]
            height, width = targetShape[2], targetShape[3]

            self.ystart = int((inputShape[2] - targetShape[2]) / 2)
            self.xstart = int((inputShape[3] - targetShape[3]) / 2)
            self.yend = self.ystart + height
            self.xend = self.xstart + width

            return [[batchSize, numChannels, height, width]]

        def forward(self, inputs):
            return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]

    # Load the model.
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")

    cv2.dnn_registerLayer('Crop', CropLayer)

    (H, W) = image.shape[:2]

    # image = cv.resize(image, (args.width, args.height))
    mean_ori = (104.00698793, 116.66876762, 122.67891434)

    inp = cv2.dnn.blobFromImage(image, scalefactor=1, size=(W, H),
                               mean=mean_ori,
                               swapRB=False, crop=False)
    net.setInput(inp)
    startTime = time()
    out = net.forward()

    print("Total time: --- %0.3f seconds ---" % (time() - startTime) + "\n")

    out = out[0, 0]
    out = cv2.resize(out, (image.shape[1], image.shape[0]))

    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)

    # generate canny edge
    image_canny = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(image_canny, 0, CannyVint)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # print(type(out))
    # print(np.max(out))
    # print(np.min(out))
    # print(out.shape)
    # print(image.shape)
    # con = np.concatenate((image, out), axis=1)
    con = out
    # cv2.imshow("HED", con)
    # cv2.imshow("canny", edges)
    # cv2.waitKey(0)
    cv2.imwrite('pic_out/' + output_folder + '/' + imageName + '_HED_out.jpg', con)
    cv2.imwrite('pic_out/' + output_folder + '/' + imageName + '_Canny_out.jpg', edges)
