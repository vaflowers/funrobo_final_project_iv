import cv2
import numpy as np
import glob

CHECKERBOARD = (9, 6) 
SQUARE_SIZE = 24e-3  

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = [] 
imgpoints = [] #

images = glob.glob('calib_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"Processed {fname}: Corners found.")
    else:
        print(f"Processed {fname}: No corners detected. Skipping.")

if len(objpoints) > 10:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    np.savez("calibration_data.npz", mtx=mtx, dist=dist)
    print("Camera Matrix:\n", mtx)
else:
    print("\nERROR: Not enough valid images. Need at least 10 where corners are visible.")