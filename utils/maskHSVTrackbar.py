"""

Pequeño script para filtros HSV y operaciones morfologicas erosion y dilatacion.

"""

import cv2
import numpy as np
import argparse


def hsv_trackbar(img):
    # Create image
    # img = cv2.imread(image_path)
    og_size = img.shape
    img = cv2.resize(img, (480,640))
    # img = img[325:625,525:875]

    # Create a window
    cv2.namedWindow("HSV")
    cv2.namedWindow("morphs_mask")

    cv2.setWindowTitle("HSV", "Filtro de Color")
    cv2.setWindowTitle("morphs_mask", "Operaciones Morfologicas sobre la máscara")
    cv2.setWindowTitle("morphs", "Operaciones Morfologicas")

    # create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar("HMin", "HSV", 0, 179, lambda x: x)  
    cv2.createTrackbar("HMax", "HSV", 0, 179, lambda x: x)
    cv2.createTrackbar("SMin", "HSV", 0, 255, lambda x: x)
    cv2.createTrackbar("SMax", "HSV", 0, 255, lambda x: x)
    cv2.createTrackbar("VMin", "HSV", 0, 255, lambda x: x)
    cv2.createTrackbar("VMax", "HSV", 0, 255, lambda x: x)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos("HMax", "HSV", 179)
    cv2.setTrackbarPos("SMax", "HSV", 255)
    cv2.setTrackbarPos("VMax", "HSV", 255)
    
    cv2.createTrackbar("Erode Iterations", "morphs", 0, 20, lambda x: x)  
    cv2.createTrackbar("Dilate Iterations", "morphs", 0, 20, lambda x: x)      
    cv2.createTrackbar("Erode Kernel Size", "morphs", 1, 20, lambda x: x)  
    cv2.createTrackbar("Dilate Kernel Size", "morphs", 1, 20, lambda x: x)  
    cv2.setTrackbarPos("Erode Iterations", "morphs", 0)
    cv2.setTrackbarPos("Dilate Iterations", "morphs", 0)
    cv2.setTrackbarPos("Erode Kernel Size", "morphs", 1)
    cv2.setTrackbarPos("Dilate Kernel Size", "morphs", 1)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = img
    waitTime = 33

    while True:

        erode_iterations = cv2.getTrackbarPos("Erode Iterations", "morphs")
        dilate_iterations = cv2.getTrackbarPos("Dilate Iterations", "morphs")
        erode_kernel_size = cv2.getTrackbarPos("Erode Kernel Size", "morphs")
        dilate_kernel_size = cv2.getTrackbarPos("Dilate Kernel Size", "morphs")
        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos("HMin", "HSV")
        sMin = cv2.getTrackbarPos("SMin", "HSV")
        vMin = cv2.getTrackbarPos("VMin", "HSV")

        hMax = cv2.getTrackbarPos("HMax", "HSV")
        sMax = cv2.getTrackbarPos("SMax", "HSV")
        vMax = cv2.getTrackbarPos("VMax", "HSV")



        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        # Print if there is a change in HSV value
        if (
            phMin != hMin
            or psMin != sMin
            or pvMin != vMin
            or phMax != hMax
            or psMax != sMax
            or pvMax != vMax
        ):
            print(
                f"(hMin = {hMin} , sMin = {sMin}, vMin = {vMin}), "
                f"(hMax = {hMax} , sMax = {sMax}, vMax = {vMax})"
                f"\nerode_iterations: {erode_iterations} dilate_iterations: {dilate_iterations}"
                f"erode_kernel_size: {erode_kernel_size} dilate_kernel_size: {dilate_kernel_size}"
            )
            phMin, psMin, pvMin = hMin, sMin, vMin
            phMax, psMax, pvMax = hMax, sMax, vMax
        # Display output image
        cv2.imshow("HSV", output)

        # Morphological operations



        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=erode_iterations)

        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=dilate_iterations)
        
        cv2.imshow("dilated mask", dilated_mask)
        output = cv2.bitwise_and(img, img, mask=dilated_mask)
        cv2.imshow("morphs_mask", output)


        eroded = cv2.erode(img, kernel, iterations=erode_iterations)
        dilated = cv2.dilate(eroded, kernel, iterations=dilate_iterations)
        cv2.imshow("morphs", dilated) 


        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(waitTime) & 0xFF == ord("q"):
            break


    
    cv2.destroyAllWindows()
    dilated_mask = cv2.resize(dilated_mask, (og_size[1], og_size[0]))

    return dilated_mask
# hsv_trackbar(f'/home/ariel/ComputerVision/VisionCherries/PolarizedImages/Artificial/Cherry_1/polarized/processed/deg_0/image_46.raw/intensity_0.png') 
