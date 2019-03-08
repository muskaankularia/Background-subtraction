import numpy as np
import numpy.linalg as linalg
import sys
import cv2
import math
import scipy.stats

cap = cv2.VideoCapture(sys.argv[1])
frame = cap.read()

def mod(x):
    if(x < 0):
        return -1 * x
    return x

def mult_normal(x, u, sigma):
    power = x-u
    inv = sigma
    fin_pow = -0.5 * power*inv*power
    exp = np.exp(fin_pow)
    denom = np.power(2 * np.pi, 1.5) * np.sqrt((sigma))
    return exp/denom



init = 15
ret, frame = cap.read()
height, width = frame.shape[:2]
samples = np.zeros((height, width, init))

while(init):
    init -= 1
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    samples[:, :, init] = gray



means_arr = np.zeros((height, width, 2))
covs_arr = np.zeros((height, width, 2))
weights_arr = np.zeros((height, width, 2))
Mahalanobis = np.zeros((height, width, 2))

counter = 0 
bg_gauss = np.zeros((height, width, 2))

T = 0.5
alpha = 0.05

cluster_n = 2
for y in range(height):
    for x in range(width):

        em = cv2.EM(cluster_n ,cv2.EM_COV_MAT_DIAGONAL)
        em.train(samples[y][x])

        means = em.getMat('means')
        covs = em.getMatVector('covs')
        weights = em.getMat('weights')


        covlist = []
        covlist.append(covs[0][0][0])
        covlist.append(covs[1][0][0])
        # print means

        means_arr[y][x] = means.T
        covs_arr[y][x] = covlist
        weights_arr[y][x] = weights[0]


counter = 0

while(1):
    ret, frame = cap.read()
    if not ret:
       break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    counter += 1

    fgmask = np.ones((height,width)) * 255
    fltSrc = gray

    bg_gauss[:,:,0] = (weights_arr[:, :, 0] > T)
    bg_gauss[:,:,1] = (weights_arr[:, :, 1] > T)

    mah_mask = np.zeros((height, width, 2), dtype = bool)
    mask = np.zeros((height, width), dtype = bool)
    x_t_one = fltSrc
    u_i_t = means_arr
    sigma_i_t = covs_arr
    
    k = 50

    for z in range(cluster_n):
        mat1 = x_t_one - u_i_t[:, :, z]
        mat2 = sigma_i_t[:, :, z]
        Mahalanobis[:, :, z] = np.sqrt(mat1*mat2*mat1) 
        mah_mask[:, :, z] = (Mahalanobis[:, :, z] < k * np.sqrt( sigma_i_t[:, :, z]))
        mask = np.logical_and(mah_mask[:, :, z], bg_gauss[:, :, z])
        print fgmask[mah_mask[:, :, z]].shape
        print height*width
        fgmask[mask] = 0


    left_out_gaussian = np.zeros((height, width, 2), dtype = bool)
    left_out_gaussian[:, :, 1] = np.logical_and(mah_mask[:,:,0], ~mah_mask[:,:,1])  
    left_out_gaussian[:, :, 0] = np.logical_and(mah_mask[:,:,1], ~mah_mask[:,:,0])  

    weights_arr[mah_mask] = (1-alpha)*weights_arr[mah_mask] + alpha
    weights_arr[left_out_gaussian] = (1-alpha)*weights_arr[left_out_gaussian]
    double_fltsrc = np.zeros((height, width, 2))
    double_fltsrc[:, :, 0] = fltSrc
    double_fltsrc[:, :, 1] = fltSrc 
    # rho = 0.01
    rho = (alpha * mult_normal(double_fltsrc, means_arr, covs_arr))[mah_mask[:, :, :]]
    means_arr[mah_mask] = (1-rho) * (means_arr[mah_mask]) + rho * double_fltsrc[mah_mask]
    mat_sub_mean = (double_fltsrc[mah_mask[:, :, :]]-means_arr[mah_mask])
    covs_arr[mah_mask] = (1-rho)*covs_arr[mah_mask] + rho*(mat_sub_mean*mat_sub_mean)

    final_mask = np.logical_or(mah_mask[:, :, 0], mah_mask[:, :, 1])
    wa1 = weights_arr[:, :, 0]
    wa2 = weights_arr[:, :, 1]
    least_probable = np.zeros((height, width, 2), dtype = bool)
    least_probable[:, :, 0] = wa1 < wa2
    least_probable[:, :, 1] = wa1 > wa2  #stores the position where 1 < 0
    least_probable[:, :, 0] = np.logical_and(least_probable[:, :, 0], ~final_mask)
    least_probable[:, :, 1] = np.logical_and(least_probable[:, :, 1], ~final_mask)
    weights_arr[least_probable] = 0.01
    means_arr[least_probable] = double_fltsrc[least_probable]
    covs_arr[least_probable] = 10000

    cv2.imshow('frame', fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


   
cap.release()
cv2.destroyAllWindows()