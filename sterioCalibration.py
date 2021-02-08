import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

images_right = glob.glob('/home/sai/Desktop/Thesis/programs/checkerboard4_cam1_orientation4.py'+'/*.png')
images_left = glob.glob('/home/sai/Desktop/Thesis/programs/checkerboard4_cam2_orientation4.py'+'/*.png')

objectpoints = []
imagepoints_left = []
imagepoints_right=[]
criteria=(cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((10*15,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:15].T.reshape(-1,2)

image_right=[]
image_left=[]
for i,filename in enumerate(images_left):
    image_left=cv2.imread(images_left[i])
    gray_left =cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    print(gray_left.shape)

    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (10,15),None)

    print(ret_left)


    if ret_left is True:
        objectpoints.append(objp)
        corners_left2 = cv2.cornerSubPix(gray_left,corners_left,(11,11),(-1,-1),criteria)
        imagepoints_left.append(corners_left2)
        image_left = cv2.drawChessboardCorners(image_left, (10,15), corners_left2,ret_left)
        cv2.imshow('img',image_left)
        cv2.waitKey(500)

objectpoints=[]

for i,filename in enumerate(images_right):
    image_right=cv2.imread(images_right[i])
    gray_right =cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (10,15),None)
    print(ret_right)
    if ret_right is True:
        objectpoints.append(objp)
        corners_right2 = cv2.cornerSubPix(gray_right,corners_right,(11,11),(-1,-1),criteria)
        imagepoints_right.append(corners_right2)
        image_right = cv2.drawChessboardCorners(image_right, (10,15), corners_right2,ret_right)
        cv2.imshow('img',image_right)
        cv2.waitKey(500)


rt, M1, d1, r1, t1 = cv2.calibrateCamera(objectpoints, imagepoints_left, gray_left.shape[::-1], None, None)


rt, M2, d2, r2, t2 = cv2.calibrateCamera(objectpoints, imagepoints_right, gray_right.shape[::-1], None, None)


flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
flags |= cv2.CALIB_USE_INTRINSIC_GUESS
flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(objectpoints, imagepoints_left,imagepoints_right,M1,d1, M2,d2,gray_left.shape,criteria=stereocalib_criteria, flags=flags)

print('Intrinsic_mtx_1', M1)
print('dist_1', d1)
print('Intrinsic_mtx_2', M2)
print('dist_2', d2)
print('R', R)
print('T', T)
print('E', E)
print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

print('')

def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image

camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),('dist2', d2),('R', R), ('T', T),
                            ('E', E), ('F', F)])



img_1 = cv2.imread('003.png')
img_2 = cv2.imread('003_1.png')
h,w = img_2.shape[:2]

new_camera_matrix1, roi1 = cv2.getOptimalNewCameraMatrix(M1,d1,(w,h),1,(w,h))
new_camera_matrix2, roi2 = cv2.getOptimalNewCameraMatrix(M1,d1,(w,h),1,(w,h))

img_1_undistorted = cv2.undistort(img_1, M1, d1, None, new_camera_matrix1)
img_2_undistorted = cv2.undistort(img_2, M2, d2, None, new_camera_matrix2)

img_1_downsampled = downsample_image(img_1_undistorted,3)
img_2_downsampled = downsample_image(img_2_undistorted,3)

win_size = 5
min_disp = -1
max_disp = 63 #min_disp * 9
num_disp = max_disp - min_disp # Needs to be divisible by 16

stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
 numDisparities = num_disp,
 blockSize = 5,
 uniquenessRatio = 5,
 speckleWindowSize = 5,
 speckleRange = 5,
 disp12MaxDiff = 1,
 P1 = 8*3*win_size**2,#8*3*win_size**2,
 P2 =32*3*win_size**2) #32*3*win_size**2)


print ("\nComputing the disparity  map...")
disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)


#Show disparity map before generating 3D cloud to verify that point cloud will be usable.
plt.imshow(disparity_map,'gray')
plt.show()



























cv2.destroyAllWindows()
