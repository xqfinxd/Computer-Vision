import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np
import math

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
    result = np.copy(img)
    [height_kernel, width_kernel] = kernel.shape
    height_extra = (height_kernel-1)/2
    width_extra = (width_kernel-1)/2
    if img.ndim == 3 :
        #RGB image
        [height_img, width_img, rgb] = img.shape
        imgcp = np.zeros((height_img+2*height_extra, width_img+2*width_extra))
        for color in range(rgb) :
            for i in range(height_img) :
                for j in range(width_img) :
                    imgcp[i+height_extra, j+width_extra] = img[i, j, color]
            for i in range(height_img): 
                for j in range(width_img) :
                    count = 0
                    for m in range(height_kernel) :
                        for n in range(width_kernel) :
                            count += imgcp[i+m, j+n]*kernel[m, n]
                    result[i, j, color] = count
        return result
    else :
        #Gray image
        [height_img, width_img] = img.shape
        imgcp = np.zeros((height_img+2*height_extra, width_img+2*width_extra))
        for i in range(height_img) :
            for j in range(width_img) :
                imgcp[i+height_extra, j+width_extra] = img[i, j]
        for i in range(height_img): 
            for j in range(width_img) :
                count = 0
                for m in range(height_kernel) :
                    for n in range(width_kernel) :
                        count += imgcp[i+m, j+n]*kernel[m, n]
                result[i, j] = count
        return result


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
    kernelcp = kernel.copy()
    [h, w] = kernel.shape
    h_index = h-1
    w_index = w-1
    for i in range(h) :
        for j in range(w) :
            kernelcp[i, j] = kernel[h_index-i, w_index-j]
    return cross_correlation_2d(img, kernelcp)



def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
    total = 0.0
    pi = 3.1415926
    height_half = (height-1)/2
    width_half = (width-1)/2
    sigma_2_2 = sigma*sigma*2.0
    result = np.zeros((height, width))

    for i in range(height) :
        for j in range(width) :
            result[i, j] = math.exp(-(((i-height_half)**2) + (j-width_half)**2) / sigma_2_2) / (sigma_2_2*pi)
            total += result[i, j]
    
    for i in range(height) :
        for j in range(width) :
            result[i, j] /= total

    return result


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END
    img_lowpass = low_pass(img, sigma, size)
    img_highpass = np.copy(img)
    if img.ndim == 3 :
        [height, width, color] = img.shape
        for c in range(color) :
            for x in range(height) :
                for y in range(width) :
                    img_highpass[x, y, c] = img[x, y, c] - img_lowpass[x, y, c]
    else :
        [height, width] = img.shape
        for x in range(height) :
            for y in range(width) :
                img_highpass[x, y] = img[x, y] - img_lowpass[x, y]

    return img_highpass

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


