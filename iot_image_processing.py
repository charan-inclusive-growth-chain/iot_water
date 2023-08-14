import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import os
import numpy as np
import cv2

# IMG_PATH = r"/home/kesavan/Desktop/work/iot_water/img_folder/photo.jpg"
def main(IMG_PATH):
   
    # read input image from your computer
    img = read_image(IMG_PATH)

    #xmin_u = 370
    #ymin_u = 350
    #xmax_u = 855
    #ymax_u = 670

    #xmin_l = 450
    #ymin_l = 810
    #xmax_l = 765
    #ymax_l = 1020
    
    xmin_l = 660
    ymin_l = 300
    xmax_l = 1010
    ymax_l = 555
    #
    xmin_u = 610
    ymin_u = 620
    xmax_u = 1010
    ymax_u = 890

    split_u1 = xmin_u + (xmax_u - xmin_u)//3
    split_u2 = xmin_u + 2*(xmax_u - xmin_u)//3

    split_l1 = xmin_l + (xmax_l - xmin_l)//3
    split_l2 = xmin_l + 2*(xmax_l - xmin_l)//3


    img = torchvision.transforms.ToPILImage()(img)
    img = np.array(img)

    ub_m_r = np.median(img[ymin_u:ymax_u,xmin_u:split_u1,0])
    ub_m_g = np.median(img[ymin_u:ymax_u,xmin_u:split_u1,1])
    ub_m_b = np.median(img[ymin_u:ymax_u,xmin_u:split_u1,2])

    uw_m_r = np.median(img[ymin_u:ymax_u,split_u1:split_u2,0])
    uw_m_g = np.median(img[ymin_u:ymax_u,split_u1:split_u2,1])
    uw_m_b = np.median(img[ymin_u:ymax_u,split_u1:split_u2,2])

    ug_m_r = np.median(img[ymin_u:ymax_u,split_u2:xmax_u,0])
    ug_m_g = np.median(img[ymin_u:ymax_u,split_u2:xmax_u,1])
    ug_m_b = np.median(img[ymin_u:ymax_u,split_u2:xmax_u,2])

    lb_m_r = np.median(img[ymin_l:ymax_l,xmin_l:split_l1,0])
    lb_m_g = np.median(img[ymin_l:ymax_l,xmin_l:split_l1,1])
    lb_m_b = np.median(img[ymin_l:ymax_l,xmin_l:split_l1,2])

    lw_m_r = np.median(img[ymin_l:ymax_l,split_l1:split_l2,0])
    lw_m_g = np.median(img[ymin_l:ymax_l,split_l1:split_l2,1])
    lw_m_b = np.median(img[ymin_l:ymax_l,split_l1:split_l2,2])

    lg_m_r = np.median(img[ymin_l:ymax_l,split_l2:xmax_l,0])
    lg_m_g = np.median(img[ymin_l:ymax_l,split_l2:xmax_l,1])
    lg_m_b = np.median(img[ymin_l:ymax_l,split_l2:xmax_l,2])


    W1 = np.array([uw_m_r, uw_m_g, uw_m_b]) # Upper White Area [R G B]
    W2 = np.array([lw_m_r, lw_m_g, lw_m_b]) # Lower White Area [R G B]

    G1 = np.array([ug_m_r, ug_m_g, ug_m_b]) # Upper Gray Area [R G B]
    G2 = np.array([lg_m_r, lg_m_g, lg_m_b]) # Lower Gray Area [R G B]

    B1 = np.array([ub_m_r, ub_m_g, ub_m_b]) # Upper Black Area [R G B]
    B2 = np.array([lb_m_r, lb_m_g, lb_m_b]) # Lower Black Area [R G B]

    depth = np.array([-5 , -15]) # Depths of upper and lower plates

    Secchi_coefficients = np.array([11.97, -0.7899])
    Turbidity_coefficients = np.array([1.32, -1.39])


    # Black area correction
    C1 = (W1-B1)/G1 # Upper Level
    C2 = (W2-B2)/G2 # Lower Level

    # Attenuation
    p = np.polyfit(depth, np.log([C1[0], C2[0]]), 1)
    K_R = p[0]*100 # Attenuation for Red
    p = np.polyfit(depth, np.log([C1[1], C2[1]]), 1)
    K_G = p[0]*100 # Attenuation for Green
    p = np.polyfit(depth, np.log([C1[2], C2[2]]), 1)
    K_B = p[0]*100 # Attenuation for Blue


    K_mean_RG = np.round(np.mean([K_R, K_G]),2) # Mean of red and green attenuations

    SD = np.round((Secchi_coefficients[0]/K_mean_RG) + Secchi_coefficients[1],2) # Secchi Depth

    Turb = np.round(Turbidity_coefficients[0]*K_R + Turbidity_coefficients[1],2) # Turbidity


    return {"K_mean_RG":K_mean_RG,"Secchi_Depth":SD,"Turbidity":Turb}

# print(main(IMG_PATH))