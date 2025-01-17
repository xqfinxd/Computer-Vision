import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN
    # raise Exception("TODO in blend.py not implemented")

    h, w = img.shape[:2]
    topleft = np.array([[0, 0, 1]]).T
    buttomleft = np.array([[0, h-1, 1]]).T
    topright = np.array([[w-1, 0, 1]]).T
    buttomright = np.array([[w-1, h-1, 1]]).T
    topleft = np.dot(M, topleft)
    topright = np.dot(M, topright)
    buttomleft = np.dot(M, buttomleft)
    buttomright = np.dot(M, buttomright)
    xlist = []
    ylist = []
    xlist.append(topright[0]/topright[2])
    xlist.append(topleft[0]/topleft[2])
    xlist.append(buttomright[0]/buttomright[2])
    xlist.append(buttomleft[0]/buttomleft[2])
    ylist.append(topright[1]/topright[2])
    ylist.append(topleft[1]/topleft[2])
    ylist.append(buttomright[1]/buttomright[2])
    ylist.append(buttomleft[1]/buttomleft[2])

    minX = min(xlist)
    maxX = max(xlist)
    minY = min(ylist)
    maxY = max(ylist)
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN
    # raise Exception("TODO in blend.py not implemented")

    maxh, maxw, ca = acc.shape
    height, width, ci = img.shape
    invM = np.linalg.inv(M)
    for i in range(maxh):
    	for j in range(maxw):
    		matp = np.array([[j, i, 1]]).T
    		matp = np.dot(invM, matp)
    		x = int(matp[0]/matp[2])
    		y = int(matp[1]/matp[2])
    		if (x<0) or (x>width-1) or (y<0) or (y>height-1):
    			continue
    		sumpix = int(img[y, x, 0])+int(img[y, x, 1])+int(img[y, x, 2])
    		if sumpix==0:
    			continue
    		minVxy = min(x, width-1-x, y, height-1-y)
    		weight = 0.0
    		if minVxy<blendWidth:
    			weight+=1.0*minVxy/blendWidth
    		else:
    			weight+=1.0
    		acc[i, j, 0] += img[y, x, 0]*weight
    		acc[i, j, 1] += img[y, x, 1]*weight
    		acc[i, j, 2] += img[y, x, 2]*weight
    		acc[i, j, 3] += weight
    
    return acc
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN
    # raise Exception("TODO in blend.py not implemented")

    height, width, channel = acc.shape
    img = np.zeros((height, width, channel-1))
    for i in range(height):
    	for j in range(width):
    		weight = acc[i, j, 3]
    		if weight == 0:
    			img[i, j, 0] = 0
    			img[i, j, 1] = 0
    			img[i, j, 2] = 0
    		else:
    			img[i, j, 0] = acc[i, j, 0]/weight
    			img[i, j, 1] = acc[i, j, 1]/weight
    			img[i, j, 2] = acc[i, j, 2]/weight

    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accHeight: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        # raise Exception("TODO in blend.py not implemented")

        nX, nY, xX, xY = imageBoundingBox(img, M)
        minX = min(nX, minX)
        maxX = max(xX, maxX)
        minY = min(nY, minY)
        maxY = max(xY, maxY)

        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    # raise Exception("TODO in blend.py not implemented")

    if is360:
    	a = -1.0*(y_final-y_init)/(x_final-x_init)
    	A[1, 0] = a
    	A[0, 2] = -0.5*width

    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )
    croppedImage = croppedImage.astype(np.uint8)
    return croppedImage

