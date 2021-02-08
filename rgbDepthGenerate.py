import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import pyrealsense2 as rs
import os
import math
from utils.rgbd_util import *
from utils.getCameraParam import *


def videotoframe(videofile,path):
    vidcap = cv2.VideoCapture(videofile)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if success==True:
            cv2.imwrite(path+"frame%d.png" % count, image)
        else:
            break
        count += 1

def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

def stereoMatcher(window_size):
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=8,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    return left_matcher,right_matcher


def depthEstimation(leftImage,rightImage,M1,M2,d1,d2,path,window_size,counter,lmbda = 80000,sigma = 1.2,visual_multiplier = 1.0):
    left_image=cv2.imread(leftImage)
    right_image=cv2.imread(rightImage)
    h,w = right_image.shape[:2]
    new_camera_matrix1, roi1 = cv2.getOptimalNewCameraMatrix(M1,d1,(w,h),1,(w,h))
    new_camera_matrix2, roi2 = cv2.getOptimalNewCameraMatrix(M2,d2,(w,h),1,(w,h))
    img_1_undistorted = cv2.undistort(left_image, M1, d1, None, new_camera_matrix1)
    img_2_undistorted = cv2.undistort(right_image, M2, d2, None, new_camera_matrix2)
    img_1_downsampled = downsample_image(img_1_undistorted,3)
    img_2_downsampled = downsample_image(img_2_undistorted,3)
    left_matcher,right_matcher=stereoMatcher(window_size)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    print('computing disparity...')
    displ = left_matcher.compute(img_1_downsampled, img_2_downsampled)  # .astype(np.float32)/16
    dispr = right_matcher.compute(img_2_downsampled, img_1_downsampled)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, img_1_downsampled, None, dispr)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    cv2.imwrite(path+str(counter)+".png", filteredImg)
    return filteredImg

def loadCamParam():
    M1=np.load('Intrinsic_leftCam.npy')
    d1=np.load('distortion_leftCam.npy')
    M2=np.load('Intrinsic_rightCam.npy')
    d2=np.load('distortion_rightCam.npy')
    return M1,M2,d1,d2

def getHHA(C, D, RD):
    missingMask = (RD == 0);
    pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C);
    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)
    angle[np.isnan(angle)] = 180
    pc[:,:,2] = np.maximum(pc[:,:,2], 100)
    I = np.zeros(pc.shape)
    I[:,:,2] = 31000/pc[:,:,2]
    I[:,:,1] = h
    I[:,:,0] = (angle + 128-90)
    I = np.rint(I)
    I[I>255] = 255
    HHA = I.astype(np.uint8)
    return HHA

def getSurfaceNormals(depthImage):
    depthImage = depthImage.astype("float64")
    normals = np.array(depthImage, dtype="float32")
    h,w,d = depthImage.shape
    for i in range(1,w-1):
        for j in range(1,h-1):
            t = np.array([i,j-1,depthImage[j-1,i,0]],dtype="float64")
            f = np.array([i-1,j,depthImage[j,i-1,0]],dtype="float64")
            c = np.array([i,j,depthImage[j,i,0]] , dtype = "float64")
            d = np.cross(f-c,t-c)
            n = d / np.sqrt((np.sum(d**2)))
            normals[j,i,:] = n

    return normals*255



def main():
    #videotoframe('/home/sai/Desktop/Thesis/programs/Cam3-part3.mp4','/home/sai/Desktop/Thesis/programs/cam1ImagesFinal/')
    #videotoframe('/home/sai/Desktop/Thesis/programs/Cam4-part4.mp4','/home/sai/Desktop/Thesis/programs/cam2ImagesFinal/')
    cam1path='/home/sai/Desktop/Thesis/programs/cam1Illumination/'
    cam2path='/home/sai/Desktop/Thesis/programs/cam2Illumination/'
    depthpath='/home/sai/Desktop/Thesis/programs/depthFinal/'
    hhadepth='/home/sai/Desktop/Thesis/programs/HHAFinal/'
    left_image=np.sort(os.listdir(cam1path))
    right_image=np.sort(os.listdir(cam2path))
    depthImage=np.sort(os.listdir(depthpath))
    window_size=5
    M1,M2,d1,d2=loadCamParam()

    for i in range(len(left_image)):
        depthImage=depthEstimation(cam1path+left_image[i],cam2path+right_image[i],M1,M2,d1,d2,depthpath,window_size,i)
        img = np.array(depthImage, dtype=np.uint16)
        img*=256
        img=img/10000
        camera_matrix = getCameraParam()
        #print(camera_matrix)
        HHA=getHHA(camera_matrix,img,img)
        #getSurfaceNormals(depthpath)
        print('Hello')
        cv2.imwrite(hhadepth+str(i)+".png",HHA)

if __name__ == '__main__':
    main()
