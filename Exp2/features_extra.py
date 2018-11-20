import math
import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()

class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def saveHarrisImage(self, harrisImage, srcImage):
        '''
        Saves a visualization of the harrisImage, by overlaying the harris
        response image as red over the srcImage.

        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        '''
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]
        harrisImage = np.zeros(srcImage.shape[:2], dtype = np.float)
        orientationImage = np.zeros(srcImage.shape[:2], dtype = np.float)

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")

        imgx = np.zeros((height, width), dtype = np.float)
        imgy = np.zeros((height, width), dtype = np.float)
        imgxx = np.zeros((height, width), dtype = np.float)
        imgxy = np.zeros((height, width), dtype = np.float)
        imgyy = np.zeros((height, width), dtype = np.float)
        ndimage.sobel(srcImage, 0, imgx)
        ndimage.sobel(srcImage, 1, imgy)
        imgxx = np.multiply(imgx, imgx)
        imgyy = np.multiply(imgy, imgy)
        imgxy = np.multiply(imgx, imgy)
        imgxx = ndimage.gaussian_filter(imgxx, 0.5)
        imgyy = ndimage.gaussian_filter(imgyy, 0.5)
        imgxy = ndimage.gaussian_filter(imgxy, 0.5)
        harris = np.zeros((2, 2), dtype = np.float)
        for x in range(0, height):
        	for y in range(0, width):
        		harris[0, 0] = imgxx[x, y]
        		harris[0, 1] = imgxy[x, y]
        		harris[1, 0] = imgxy[x, y]
        		harris[1, 1] = imgyy[x, y]
        		harrisImage[x, y] = np.linalg.det(harris) - 0.1 * np.square(np.trace(harris))
        orientationImage = np.arctan2(imgx, imgy)*180.0/np.pi
        
        # TODO-BLOCK-END

        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")

        maximg = ndimage.maximum_filter(harrisImage, [7, 7])
        minimg = ndimage.minimum_filter(harrisImage, [7, 7])
        height, width = harrisImage.shape[:2]
        for i in range(0, height):
            for j in range(0, width):
                destImage[i, j] = (maximg[i, j] == harrisImage[i, j]) and (maximg[i, j] != minimg[i, j])
                #destImage[i, j] = (maximg[i, j] == harrisImage[i, j])

        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                # raise Exception("TODO in features.py not implemented")

                f.size = 10
                f.pt = (x, y)
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]
                
                # TODO-BLOCK-END

                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        return detector.detect(image,None)

## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        img_border = cv2.copyMakeBorder(grayImage, 2, 2, 2, 2, cv2.BORDER_REFLECT)
        for i, f in enumerate(keypoints):
            x, y = f.pt
            x, y = int(x), int(y)

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # TODO-BLOCK-BEGIN
            # raise Exception("TODO in features.py not implemented")

            count = 0
            for h in range(y, y+5):
                for w in range(x, x+5):
                    desc[i, count] = img_border[h, w]
                    count += 1

            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN
            # raise Exception("TODO in features.py not implemented")
            '''
            mat:(2x3)*(3x1)
            1 step:(3x3)*(3x1)=(3x1)    --transform1
            2 step:(3x3)*(3x1)=(3x1)    --rotate
            3 step:(3x3)*(3x1)=(3x1)    --scale
            4 step:(3x2)*(3x1)=(3x1)    --transform2
            >>>(3x3)*(3x3)*(3x3)*(3x2)=(3x2)
            >>>T(3x2)=(2x3) --result
            '''
            x, y = f.pt
            transMx1 = np.array([[1, 0, 0], [0, 1, 0], [-x, -y, 1]])
            angle = math.radians(360.0 - f.angle)
            rotateMx = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
            scaleMx = np.array([[0.2, 0.0, 0], [0.0, 0.2, 0], [0, 0, 1]])
            transMx2 = np.array([[1, 0], [0, 1], [4, 4]])
            tMx = np.dot(transMx1, rotateMx)
            tMx = np.dot(tMx, scaleMx)
            tMx = np.dot(tMx, transMx2)
            transMx = np.transpose(tMx)

            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            # raise Exception("TODO in features.py not implemented")

            st = np.std(destImage)
            count = 0
            if(st < 1.0e-5):
                for h in range(0, windowSize):
                    for w in range(0, windowSize):
                        desc[i, count] = 0.0
                        count+=1
            else:
                me = np.mean(destImage)
                for h in range(0, windowSize):
                    for w in range(0, windowSize):
                        desc[i, count] = (destImage[h, w]-me) / st
                        count+=1

            # TODO-BLOCK-END
        print(desc.shape)
        return desc

    def describeFeaturesWithSIFT(self, image, fill = None):
        image = image.astype(np.float32)
        image /= 255.
        windowSize = 8
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#           grayImage = ndimage.gaussian_filter(grayImage, 0.5)
        height, width = grayImage.shape
        scaleRate = [0.25, 0.5, 1.0, 2.0, 4.0]
        imgList = []
        harrisList = []
        orienList = []
        maxboolList = []
        keypoints = []
        keypointsScale = []
        for i, s in enumerate(scaleRate):
            newImg = cv2.resize(grayImage, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            imgList.append(newImg)
        for i, img in enumerate(imgList): 
            hkd = HarrisKeypointDetector()
            hv, orie = hkd.computeHarrisValues(img)
            mb = hkd.computeLocalMaxima(hv)
            harrisList.append(hv)
            orienList.append(orie)
            maxboolList.append(mb)
        for h in range(0, height):
            for w in range(0, width):
                tmpHarris = []
                tmpIdx = []
                tmpOri = []
                for i, s in enumerate(scaleRate):
                    scaleMat = np.array([[s, 0], [0, s]], dtype = np.int)
                    point = np.array([w, h], dtype = np.int)
                    point = np.dot(point, scaleMat)
                    y, x = point
                    if maxboolList[i][y, x]:
                        tmpHarris.append(harrisList[i][y, x])
                        tmpOri.append(orienList[i][y, x])
                        tmpIdx.append(s)
                if len(tmpHarris) > 0:
                    keypoint = cv2.KeyPoint()
                    keypoint.size = 10
                    Idx = np.argmax(tmpHarris)
                    keypoint.response = tmpHarris[Idx]
                    keypoint.pt = (w, h)
                    keypoint.angle = tmpOri[Idx]
                    keypoints.append(keypoint)
                    keypointsScale.append(tmpIdx[Idx])

        desc = np.zeros((len(keypoints), windowSize * windowSize))

        for i, f in enumerate(keypoints):
            transMx = np.zeros((2, 3))
            x, y = f.pt
            transMx1 = np.array([[1, 0, 0], [0, 1, 0], [-x, -y, 1]])
            angle = math.radians(360.0 - f.angle)
            rotateMx = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
            ###CHANGED HERE : scaleMx
            scaleMx = np.array([[0.2*keypointsScale[i], 0.0, 0], [0.0, 0.2*keypointsScale[i], 0], [0, 0, 1]])  
            transMx2 = np.array([[1, 0], [0, 1], [4, 4]])
            tMx = np.dot(transMx1, rotateMx)
            tMx = np.dot(tMx, scaleMx)
            tMx = np.dot(tMx, transMx2)
            transMx = np.transpose(tMx)

            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            st = np.std(destImage)
            count = 0
            if(st < 1.0e-5):
                for h in range(0, windowSize):
                    for w in range(0, windowSize):
                        desc[i, count] = 0.0
                        count+=1
            else:
                me = np.mean(destImage)
                for h in range(0, windowSize):
                    for w in range(0, windowSize):
                        desc[i, count] = (destImage[h, w]-me) / st
                        count+=1
        print(desc.shape)
        return desc
class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")
        des1num = desc1.shape[0]
        des2num = desc2.shape[0]
        dimenum = desc1.shape[1]
        for i in range(0, des1num):
            mtc = cv2.DMatch()
            total = []
            for j in range(0, des2num):
                s = 0
                for c in range(0, dimenum):
                    s += (desc1[i, c] - desc2[j, c])**2
                total.append(s)
            mtc.queryIdx = i
            mtc.trainIdx = np.argmin(total)
            mtc.distance = total[mtc.trainIdx]
            matches.append(mtc)
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")

        des1num = desc1.shape[0]
        des2num = desc2.shape[0]
        dimenum = desc1.shape[1]
        for i in range(0, des1num):
            mtc = cv2.DMatch()
            total = []
            for j in range(0, des2num):
                s = 0
                for c in range(0, dimenum):
                    s += (desc1[i, c] - desc2[j, c])**2
                total.append(s)
            firIdx = np.argmin(total)
            firMin = total[firIdx]
            total[firIdx] = max(total)
            secIdx = np.argmin(total)
            secMin = total[secIdx]
            total[firIdx] = firMin
            dis = firMin/secMin
            mtc.queryIdx = i
            mtc.trainIdx = firIdx
            mtc.distance = dis
            matches.append(mtc)

        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))

