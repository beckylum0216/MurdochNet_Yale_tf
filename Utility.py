import numpy as np
import os
import cv2
from PIL import Image

import ImageHeader

class Utility(object):

    def readGIFFile(self, filename, imgHdr = ImageHeader.ImageHeader()):
        frame = Image.open(filename).convert('L')
        npImage = np.array(frame)

        # below is for direct input into tensorflow
        # imgInput = np.true_divide(np.array(frame),255)

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

    def SaveImageFile(self, filepath, targetImg):
        cv2.imwrite(filepath, targetImg)

