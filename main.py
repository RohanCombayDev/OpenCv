import cv2
import numpy as np

# kernel = np.ones((5,5),np.uint8)

# reading_image
# img = cv2.imread("xyz.jpg")
# cv2.imshow("OUTPUT",img)

# cv2.waitKey(0) #0 means infinite time

# reading_video
# cap=cv2.VideoCapture("video1.mp4")
# while True:
#    success, img=cap.read()
#    cv2.imshow("Video",img)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
# using webcam
# webcam= cv2.VideoCapture(0)
# webcam.set(3,640) #width
# webcam.set(4,480) #height syntax:id,length
# webcam.set(10,1000) #brightness

# while True:
#    success, img=webcam.read()
#    cv2.imshow("Video",img)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

# BasicFunctions
# converting image to another color
# img = cv2.imread('xyz.jpg')
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("GrayImage",imgGray)
# cv2.waitKey(0)

# blurring image
# imgBlur = cv2.GaussianBlur(img,(7,7),0)
# cv2.imshow("BlurImage",imgBlur)
# cv2.waitKey(0)

# EdgeDetector
# imgCanny = cv2.Canny(img,150,200) #edging image
# imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)
# imgErode = cv2.erode(imgCanny,kernel,iterations=1)
# cv2.imshow("EdgeImage",imgCanny)
# cv2.imshow("DialtionImage",imgDialation) #increasing thickness of edges
# cv2.imshow("ErodedIMgae",imgErode)
# cv2.waitKey(0)

# resizing and cropping
# print(img.shape)
# imgresize = cv2.resize(img,(300,200)) #width,height
# cv2.imshow("resized",imgresize)
# cv2.waitKey(0)
# print(imgresize.shape)

# cropping image
# imgcropped = img[0:200,200:500]#height,width
# cv2.imshow("cropped",imgcropped)
# cv2.waitKey(0)

# drawing shapes
# img=np.zeros((512,512,3),np.uint8)
# img[:]=255,0,0   # : gives limit
# img[200:300,300:500] = 255,0,0 #specific part
# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3) #drawing line
# text in image
# cv2.putText(img,'OpenCV',(300,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)
# cv2.imshow("IMage",img)
# cv2.waitKey(0)
# print(img.shape)

# color detection
# def empty(a):
#    pass

# cv2.namedWindow("Trackbar")
# cv2.resizeWindow("Trackbar",640,240)
# cv2.createTrackbar("Hue Min","Trackbar",0,179,empty)
# cv2.createTrackbar("Hue Max","Trackbar",179,179,empty)
# cv2.createTrackbar("Sat Min","Trackbar",0,255,empty)
# cv2.createTrackbar("Sat Max","Trackbar",255,255,empty)
# cv2.createTrackbar("Val Min","Trackbar",0,255,empty)
# cv2.createTrackbar("Val Max","Trackbar",255,179,empty)

# while True:
#    img = cv2.imread("xyz.jpg")
#    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#    h_min = cv2.getTrackbarPos("Hue Min","Trackbar")
#    h_max = cv2.getTrackbarPos("Hue Max", "Trackbar")
#    s_min = cv2.getTrackbarPos("Sat Min", "Trackbar")
#    s_max = cv2.getTrackbarPos("Sat Max", "Trackbar")
#    v_min = cv2.getTrackbarPos("Val Min", "Trackbar")
#    v_max = cv2.getTrackbarPos("Val Max", "Trackbar")
#    print(h_min,h_max,s_min,s_max,v_min,v_max)
#    lower=np.array([h_min,s_min,v_min])
#    upper=np.array([h_max,s_max,v_max])
#    mask=cv2.inRange(imgHSV,lower,upper)
#    imgresult = cv2.bitwise_and(img,img,mask=mask)
#    cv2.imshow("Original",img)
#    cv2.imshow("HSV",imgHSV)
#    cv2.imshow("Mask", mask)
#    cv2.imshow("result",imgresult)
#    cv2.waitKey(1)

# counter/shape detection
img = cv2.imread('shape.png')
imgContour = img.copy()


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        print(len(approx))
        objcor = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        if objcor == 3:
            ObjectType = "Tri"
        elif objcor == 4:
            aspRatio = w / float(h)
            if aspRatio > 0.95 and aspRatio < 1.05:
                ObjectType = "Square"
            else:
                ObjectType = "Rectangle"
        elif objcor > 7:
            ObjectType = "Circle"
        else:
            ObjectType = "None"

        cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(imgContour, ObjectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 0, 0), 2)


# convert to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
getContours(imgCanny)
cv2.imshow("Canny", imgCanny)
cv2.imshow("Contour", imgContour)
# cv2.imshow("Shapes",img)
# cv2.imshow("Gray",imgGray)
# cv2.imshow("Blur",imgBlur)

cv2.waitKey(0)
