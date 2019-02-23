import numpy as np
import cv2
from PIL import  Image
import ImageHeader
import Utility
import os
from ImageSet import ImageSet
from Monolithic import NeuralNet
from Preprocessor import ProcessImage

def main():

    imgHdr = ImageHeader.ImageHeader()
    imgHdr.maxImages = 165
    imgHdr.imgWidth = 320
    imgHdr.imgHeight = 243

    ut = Utility.Utility()
    dirname = "./yale/yalefaces"
    filename = ut.readFileLabel(dirname)

    for ii in range(len(filename)):
        img = ut.readGIFFile("./Yale/yalefaces/" + filename[ii], imgHdr)
        imgprocess = ProcessImage(imgHdr.imgWidth, imgHdr.imgHeight, img)
        imgprocess.DetectFace()
        croppedImg = imgprocess.CropImage()
        scaledImg = imgprocess.ScaleImage(croppedImg, 160, 160)
        gaborImg = imgprocess.ApplyGaborFilter(scaledImg)
        filePath = "./Yale/gaborfaces/" + filename[ii] + ".jpg"
        ut.SaveImageFile(filePath, gaborImg)

    # print("filenames: ", filename)
    dirname = "./Yale/yalefaces/"
    gaborname = ut.readFileLabel(dirname)
    file70, file20, file10 = ut.McCallRule(gaborname)

    train_images = ImageSet()
    #print(test_images.GetImageCount())
    train_images.LoadFromList(file70, 'Yale/yalefaces')
    #print(test_images.GetImageCount())
    imgs = train_images.GetImageRange(range(0, train_images.GetImageCount()))

    test_images = ImageSet()
    test_images.LoadFromList(file20, 'Yale/yalefaces')
    testImg = test_images.GetImageRange(range(0, test_images.GetImageCount()))


    save_base = os.path.join('.', 'saves')
    nt = NeuralNet(imgs['data'].shape[1],
                   train_images.GetUniqueLabelCount())
    for epoch in [2400]:
        save_path = os.path.join(save_base, f'save_{str(epoch).zfill(6)}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        saved_path = nt.CheckNet( testImg, imgs, epoch, os.path.join(save_path, 'yale'))

        print(saved_path)




    # meta = './saves/save_001600/yale.meta'
    # path = './saves/save_001600'
    # nt = NeuralNet(testImg['data'].shape[1], test_images.GetUniqueLabelCount())
    # #nt.InitialiseSession()
    # nt.RestoreState(meta)
    # nt.TestNet(testImg, path)

    # cv2.imshow("GIF Image", img)
    # cv2.imshow("Filtered Image", filtered_img)
    # cv2.waitKey()



if __name__ == '__main__':
    main()
