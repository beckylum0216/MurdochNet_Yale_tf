import numpy as np
import cv2
from PIL import  Image
import ImageHeader
import Utility
import os
import random
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
        croppedPath = "./Yale/croppedfaces/" + filename[ii] + ".jpg"
        ut.SaveImageFile(croppedPath, croppedImg)
        scaledImg = imgprocess.ScaleImage(croppedImg, 160, 160)
        gaborImg = imgprocess.ApplyGaborFilter(scaledImg)
        filePath = "./Yale/gaborfaces/" + filename[ii] + ".jpg"
        ut.SaveImageFile(filePath, gaborImg)

    # print("filenames: ", filename)
    dirname = "./Yale/gaborfaces/"
    gaborname = ut.readFileLabel(dirname)
    # random.shuffle(gaborname)
    randomImg = ut.RandomImg(gaborname)
    file70, file20, file10 = ut.McCallRuleWrap(gaborname)
    pareto90, pareto10 = ut.ParetoRule(gaborname)


    train_images = ImageSet()
    #print(test_images.GetImageCount())
    train_images.LoadFromList(pareto90, 'Yale/gaborfaces')
    #print(test_images.GetImageCount())
    imgs = train_images.GetRandomImages(range(0, train_images.GetImageCount()),15)

    test_images = ImageSet()
    test_images.LoadFromList(pareto10, 'Yale/gaborfaces')
    testImg = test_images.GetRandomImages(range(0, test_images.GetImageCount()),15)

    save_base = os.path.join('.', 'saves')
    nt = NeuralNet(imgs['data'].shape[1],
                   train_images.GetUniqueLabelCount())

    for epoch in [1600, 1200, 800, 400]:
        matrixPath = "confusion/confusion" + str(epoch) + "-pareto10-11-gabor.txt"
        reportPath = "classification/classreport" + str(epoch) + "-pareto10-11-gabor.txt"
        save_path = os.path.join(save_base, f'save_{str(epoch).zfill(6)}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        saved_path = nt.CheckNet( testImg, imgs, epoch, os.path.join(save_path, 'yale'), matrixPath, reportPath)

        print(saved_path)




    # meta = './saves/save_001600/yale.meta'
    # path = './saves/save_001600'
    # nt = NeuralNet(testImg['data'].shape[1], test_images.GetUniqueLabelCount())
    # #nt.InitialiseSession()
    # nt.RestoreState(meta)
    # nt.TestNet(testImg, path)

    # cv2.imshow("GIF Image", img)


    ut.DisplayImage(pareto10, "./Yale/croppedfaces")

    # print("Display with overlay")
    #ut.DisplayWithOverlay(file20[5], "./Yale/croppedfaces", "TARGA.tga", "./assets/logos")


if __name__ == '__main__':
    main()
