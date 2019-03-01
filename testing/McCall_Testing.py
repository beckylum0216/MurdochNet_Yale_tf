import unittest
import Utility
import os
from ImageSet import  ImageSet
from Monolithic import NeuralNet

def test_McCall(self):

    ut = Utility.Utility()
    # print("filenames: ", filename)
    dirname = "../Yale/gaborfaces/"
    gaborname = ut.readFileLabel(dirname)
    file70, file20, file10 = ut.McCallRuleWrap(gaborname)
    pareto90, pareto10 = ut.ParetoRule(gaborname)
    randomImg = ut.RandomImg(gaborname)

    train_images = ImageSet()
    # print(test_images.GetImageCount())
    train_images.LoadFromList(pareto90, '../Yale/gaborfaces')
    # print(test_images.GetImageCount())
    imgs = train_images.GetImageRange(range(0, train_images.GetImageCount()))

    test_images = ImageSet()
    test_images.LoadFromList(randomImg, '../Yale/gaborfaces')
    testImg = test_images.GetImageRange(range(0, test_images.GetImageCount()))

    save_base = os.path.join('.', 'saves')
    nt = NeuralNet(imgs['data'].shape[1],
                   train_images.GetUniqueLabelCount())



    for epoch in [1600]:
        matrixPath = "../confusion/confusion"+ str(epoch)+".txt"
        reportPath = "../classification/classreport"+str(epoch)+".txt"

        save_path = os.path.join(save_base, f'save_{str(epoch).zfill(6)}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        saved_path = nt.CheckNet(testImg, imgs, epoch, os.path.join(save_path, 'yale'), matrixPath, reportPath)

        print(saved_path)

    self.assertTrue(saved_path)