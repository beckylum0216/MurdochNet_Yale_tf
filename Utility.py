import numpy as np
import os
import cv2
import random
from PIL import Image

import ImageHeader

class Utility(object):

    def readGIFFile(self, filename, imgHdr = ImageHeader.ImageHeader()):
        frame = Image.open(filename).convert('L')
        npImage = np.array(frame)

        # below is for direct input into tensorflow
        # imgInput = np.true_divide(np.array(frame),255)
        frame.close()
        return npImage

    def readFileLabel(self, dirname):
        self.filelist = []
        for root, dirs, files in os.walk(dirname):
            for filename in files:
                self.filelist.append(filename)

        return self.filelist

    def McCallRule(self, fileList = []):
        col = 11
        fileMatrix = [[0] * col for x in range(15)]
        numOfFile = len(fileList)
        mccall70 = []
        mccall20 = []
        mccall10 = []

        for ii in range(165):
            jj = ii % 15
            kk = ii % 11
            fileMatrix[jj][kk] = fileList[ii]
            print("file list: ", fileMatrix[jj][kk])

        for aa in range(15):
            for bb in range(8):
                mccall70.append(fileMatrix[aa][bb])

            for cc in range(8,10):
                mccall20.append((fileMatrix[aa][cc]))

            for dd in range(10,11):
                mccall10.append(fileMatrix[aa][dd])

        return mccall70, mccall20, mccall10

    def McCallRuleWrap(self, fileList = []):
        col = 11
        fileMatrix = [[0] * col for x in range(15)]
        numOfFile = len(fileList)
        mccall70 = []
        mccall20 = []
        mccall10 = []

        for ii in range(165):
            jj = ii % 15
            kk = ii % 11
            fileMatrix[jj][kk] = fileList[ii]
            print("file list: ", fileMatrix[jj][kk])

        for aa in range(15):
            for bb in range(9, 11):
                mccall70.append(fileMatrix[aa][bb])

            for ee in range(0, 6):
                mccall70.append(fileMatrix[aa][ee])

            for cc in range(6, 8):
                mccall20.append((fileMatrix[aa][cc]))

            for dd in range(8, 9):
                mccall10.append(fileMatrix[aa][dd])


        return mccall70, mccall20, mccall10

    def ParetoRule(self, fileList = []):
        col = 11
        fileMatrix = [[0] * col for x in range(15)]
        pareto90 = []
        pareto10 = []

        for ii in range(165):
            jj = ii % 15
            kk = ii % 11
            fileMatrix[jj][kk] = fileList[ii]
            print("file list: ", fileMatrix[jj][kk])

        for aa in range(15):
            for bb in range(5,11):
                pareto90.append(fileMatrix[aa][bb])

            for dd in range(0,4):
                pareto90.append((fileMatrix[aa][dd]))

            for cc in range(4,5):
                pareto10.append((fileMatrix[aa][cc]))

        return pareto90, pareto10


    def RandomImg(self, filelist=[]):
        randomImg = []
        for ii in range(15):
            imgIndex = random.randrange(165)
            randomImg.append(filelist[imgIndex])

        return  randomImg

    def SaveImageFile(self, filepath, targetImg):
        cv2.imwrite(filepath, targetImg)

    def DisplayImage(self, targetList, dirname):
        for filename in targetList:
            frame = Image.open(dirname +'/'+ filename)
            npImage = np.array(frame)
            cv2.imshow("Labeled image", npImage)
            cv2.waitKey()
            cv2.destroyAllWindows()

    def SaveConfusionMatrix(self, targetMatrix, filepath):
        matrixFile = open(filepath, "w")

        matrixFile.writelines(targetMatrix)

        matrixFile.close()

    def SaveClassification(self, targetReport, filepath):

        classFile = open(filepath, "w")
        classFile.writelines(targetReport)
        classFile.close()