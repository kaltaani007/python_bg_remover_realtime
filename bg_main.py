import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

import os
cap = cv2.VideoCapture(0)

cap.set(3 , 640)
cap.set(4 , 480)

#cap.set(cv2.CAP_PROP_FPS , 60 )

# creating a list of images to be used as background

listImg = os.listdir ("Images")
print(listImg)

imgList = []
for i in listImg:
    img = cv2.imread("Images/" + i )
    imgList.append(img)
print(len(imgList))

# testbg
bg = cv2.imread("Images/bg1.jpg")

imgIndex  = 0


fpsReader = cvzone.FPS()

seg = SelfiSegmentation()



while True:
    ret , frame = cap.read()
    frame = cv2.flip(frame , 1)
    # (255 , 0 , 0 ) ,
    f_out = seg.removeBG(frame , imgList[imgIndex] ,  threshold = 0.8)

    imgStacked = cvzone.stackImages( [ frame , f_out ] , 2 , 1 )
    f , imgStacked = fpsReader.update(imgStacked)

    #cv2.imshow("Frame window" , frame)
    #cv2.imshow("F_out window", f_out)

    cv2.imshow("Images" , imgStacked)

    print(imgIndex)


    if cv2.waitKey(1) == ord('q'):
        break

    # controlling images with keys
    if cv2.waitKey(1) == ord('a'):
        if imgIndex > 0 :
            imgIndex -= 1

    if cv2.waitKey(1) == ord('d'):

        if imgIndex < len(imgList)    :
            imgIndex += 1







cv2.destroyAllWindows()
cv2.release()