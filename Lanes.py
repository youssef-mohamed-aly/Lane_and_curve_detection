import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    #blur = cv.GaussianBlur(gray, (5, 5),0)  # (5,5) kernal/ window size  0 for kernal diviation along x axis
    #canny applies blur internally

    canny = cv.Canny(gray, 50, 150)  # 50 min thresh 150 max thresh
    return canny


def region_of_intrest(image):
    height = image.shape[0]
    ####                200 on x axis 700 Y axis , 1100 x axis , triangle head  550 x axis , 250 y axis
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    #fill the new empty image with the white color at the crosponding points
    cv.fillPoly(mask, polygons, 255)
    # show the intrested region only
    masked_image = cv.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for  x1, y1, x2, y2 in lines:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else :
            right_fit.append((slope, intercept))
    left_fit_average =np.average(left_fit,axis =0)
    right_fit_average =np.average(right_fit,axis =0)
    left_line =make_coordinates(image,left_fit_average)
    right_line= make_coordinates(image , right_fit_average)

    return np.array([left_line , right_line])


def make_coordinates(image , line_parameters):
    slope, intercept = line_parameters
    y1= image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope) 
    x2 = int((y2-intercept)/slope)   
    return np.array([x1,y1,x2,y2])



# image =cv.imread("Image_project/curve/test_image.jpg")
# lane_image= np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_intrest(canny_image)
# lines = cv.HoughLinesP(cropped_image, 2 , np.pi/180 , 100 , np.array([]), minLineLength = 40, maxLineGap= 5 )
# average_lines = average_slope_intercept(lane_image,lines)
# line_image = display_lines(lane_image, average_lines)
# combo_image = cv.addWeighted(lane_image, 0.8 , line_image , 1 ,1)
# cv.imshow("result", combo_image)
# cv.waitKey(0)

cap = cv.VideoCapture("Image_project/curve/Untitled.mp4")
while(cap.isOpened()):
    flag, frame = cap.read()

    canny_image = canny(frame)
    # checking on X Y axis for region of intrest
    # plt.imshow(canny_image)
    # plt.show()

    cropped_image = region_of_intrest(canny_image)
 #######################################                 thresh 
    lines = cv.HoughLinesP(cropped_image, 2 , np.pi/180 , 100 , np.array([]), minLineLength = 40, maxLineGap= 5 )

    average_lines = average_slope_intercept(frame,lines)

    line_image = display_lines(frame, average_lines)

# multi every element in the image with 0.8 in order to increase it;s intensity to be more darker, second one will be multi by 1 , 
# gamma is 1 in order not to affect anything
 
    combo_image = cv.addWeighted(frame, 0.8 , line_image , 1 ,1)

    cv.imshow("result", combo_image)
    
    if cv.waitKey(3) ==ord('q'):
        break

cap.release()

cv.destroyAllWindows()