import numpy as np
import cv2
from PIL import  Image
import ImageHeader
import Utility
import os
from NeuralTrainer import NeuralTrainer
from ImageSet import ImageSet
from Monolithic import NeuralNet

def main():

    face_cascade = cv2.CascadeClassifier("./haar_cascade/haarcascade_frontalface_default.xml")

    imgHdr = ImageHeader.ImageHeader()
    imgHdr.maxImages = 165
    imgHdr.imgWidth = 320
    imgHdr.imgHeight = 243

    ut = Utility.Utility()
    dirname = "./yale/yalefaces"
    filename = ut.readFileLabel(dirname)
    print("filenames: ", filename)
    file70, file20, file10 = ut.McCallRule(filename)

    print("sample 70:", file70)

    for ii in range(len(file70)):
        image = ut.readGIFFile("./Yale/yalefaces/" + file70[ii], imgHdr)

    # faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # for x, y, w, h in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    #
    # gabor_filter = cv2.getGaborKernel((imgHdr.imgWidth, imgHdr.imgHeight), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    # filtered_img = cv2.filter2D(img, cv2.CV_8UC3, gabor_filter)



    # train_images = ImageSet()
    # print(test_images.GetImageCount())
    # train_images.LoadFromList(file70, 'Yale/yalefaces')
    # print(test_images.GetImageCount())
    # imgs = train_images.GetImageRange(range(0, train_images.GetImageCount()))


    # save_base = os.path.join('.', 'saves')
    # nt = NeuralNet(imgs['data'].shape[1],
    #                train_images.GetUniqueLabelCount())
    # for epoch in [100, 200, 400, 800, 1600, 2400]:
    #     save_path = os.path.join(save_base, f'save_{str(epoch).zfill(6)}')
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #
    #     saved_path = nt.TrainNet(imgs, epoch, os.path.join(save_path, 'yale'))
    #
    #     print(saved_path)

    test_images = ImageSet()
    test_images.LoadFromList(file20, 'Yale/yalefaces')
    testImg = test_images.GetImageRange(range(0, test_images.GetImageCount()))


    meta = './saves/save_001600/yale.meta'
    path = './saves/save_001600'
    nt = NeuralNet(testImg['data'].shape[1], test_images.GetUniqueLabelCount())
    #nt.InitialiseSession()
    nt.RestoreState(meta)
    nt.TestNet(testImg, path)

    # cv2.imshow("GIF Image", img)
    # cv2.imshow("Filtered Image", filtered_img)
    # cv2.waitKey()



if __name__ == '__main__':
    main()
