import cv2
import numpy as np
import math
from .Utils import *

class EdgeDetection:
    @staticmethod
    def detectBySobel(src, ratio=15, image_name=""):
        if len(src.shape) > 2:
            print("Source image must be grayscale")
            return -1

        # Create Sobel filters
        filterX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        filterY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

        # Calculate gradients
        gradX = cv2.filter2D(src, -1, filterX)
        gradY = cv2.filter2D(src, -1, filterY)

        # Calculate gradient magnitude
        grad = cv2.magnitude(gradX.astype(np.float64), gradY.astype(np.float64))

        # Apply threshold
        _, dst = cv2.threshold(grad, ratio * grad.max() / 100, 255, cv2.THRESH_BINARY)

        if image_name:
            cv2.imwrite(image_name + "_sobel.jpg", dst)
            
        dst = cv2.normalize(dst, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return dst

    @staticmethod
    def detectByPrewitt(src, ratio=15, image_name=""):
        if len(src.shape) > 2:
            print("Source image must be grayscale")
            return -1

        # Create Prewitt filters
        filterX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        filterY = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)

        # Calculate gradients
        gradX = cv2.filter2D(src, -1, filterX)
        gradY = cv2.filter2D(src, -1, filterY)

        # Calculate gradient magnitude
        grad = cv2.magnitude(gradX.astype(np.float64), gradY.astype(np.float64))

        # Apply threshold
        _, dst = cv2.threshold(grad, ratio * grad.max() / 100, 255, cv2.THRESH_BINARY)

        if image_name:
            cv2.imwrite(image_name + "_prewitt.jpg", dst)
                
        dst = cv2.normalize(dst, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return dst

    @staticmethod
    def zero_cross_detection(image):
        z_c_image = np.zeros(image.shape)

        for i in range(0,image.shape[0]-1):
            for j in range(0,image.shape[1]-1):
                if image[i][j]>0:
                    if image[i+1][j] < 0 or image[i+1][j+1] < 0 or image[i][j+1] < 0:
                        z_c_image[i,j] = 1
                elif image[i][j] < 0:
                    if image[i+1][j] > 0 or image[i+1][j+1] > 0 or image[i][j+1] > 0:
                        z_c_image[i,j] = 1
        return z_c_image

    @staticmethod
    def detectByLaplacian(src, ratio=10, image_name=""):
        if len(src.shape) > 2:
            print("Source image must be grayscale")
            return -1

        # Create Laplacian filter
        LoG = cv2.Laplacian(src, cv2.CV_64F)

        # Apply zero-crossing
        # dst = EdgeDetection.zero_cross_detection(LoG)

        # if image_name:
        #     cv2.imwrite(image_name + "_laplace.jpg", dst)
        
        dst = cv2.normalize(LoG, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return dst

    @staticmethod
    def detectByCanny(src, low_threshold=50.0, high_threshold=100.0, image_name=""):
        if len(src.shape) > 2:
            print("Source image must be grayscale")
            return -1

        # # Calculate gradients using Sobel
        # gradX = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        # gradY = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
        # grad = cv2.magnitude(gradX.astype(np.float64), gradY.astype(np.float64))

        # # Calculate edge directions
        # dirs = CalEdgeDirections(gradX, gradY)

        # # Perform non-maximum suppression
        # nms = NonMaxSuppression(grad, dirs)

        # # Apply hysteresis thresholding
        # dst = Hysteresis(nms, low_threshold, high_threshold)

        # if image_name:
        #     cv2.imwrite(image_name + "_canny.jpg", dst)

        # # Use OpenCV's Canny as reference
        canny_dst = cv2.Canny(src, low_threshold, high_threshold)
        
        # if image_name:
        #     cv2.imwrite(image_name + "_canny_opencv.jpg", canny_dst)

        dst = cv2.normalize(canny_dst, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return dst