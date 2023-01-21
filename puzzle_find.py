from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
import argparse

#image=cv2.imread("sudoku1.webp") #--> This is the puzzle which can be supplied for testing


def find_puzzle(image, debug=False):
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray, (7,7), 3)
    thresh=cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh=cv2.bitwise_not(thresh)
    
    if debug:
        cv2.imshow("Puzzle Thresh",thresh)
        cv2.waitKey(0) 

    cnts=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    cnts=sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzleC=None
    for c in cnts:
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*peri, True)
        if len(approx)==4:
            puzzleC=approx
            break
    if puzzleC.all()==None:
        raise Exception("Could not find outline of Sudoku Puzzle")
    if debug:
        output=image.copy()
        cv2.drawContours(output, [puzzleC], -1, (0,255,0), 2)
        cv2.imshow("Puzzle outline",output)
        cv2.waitKey(0)
    



    puzzle=four_point_transform(image,puzzleC.reshape(4,2))
    warped=four_point_transform(gray, puzzleC.reshape(4,2))

    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    
    return puzzle,warped
    
#find_puzzle(image,debug=True) 

def extract_digit(cell, debug=False):
    thresh=cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV| cv2.THRESH_OTSU)[1]
    thresh=clear_border(thresh)
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
    cnts=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    if len(cnts)==0:
        return None
    c=max(cnts,key=cv2.contourArea)
    mask=np.zeros(thresh.shape,dtype="uint8")
    cv2.drawContours(mask,[c],-1,255,-1)

    (h,w)=thresh.shape
    filledPercent=cv2.countNonZero(mask)/float(h*w)
    if filledPercent<0.0003:
        return None
    digit=cv2.bitwise_and(thresh,thresh, mask=mask)
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitkey(0)
    return digit 
    '''
ap=argparse.ArgumentParser()
ap.add_argument("-d","--debug",type=int,default=-1, help="Wheteher or not we are visualizing each step of the pipeline")
args=vars(ap.parse_args())
(puzzleImage,warped)=find_puzzle(image,debug=args["debug"]>0)
stepX=warped.shape[1]//9
stepY=warped.shape[0]//9

for y in range(0,9):
    row=[]
    for x in range(0,9):
        startX=x*stepX
        startY=y*stepY
        endX=(x+1)*stepX
        endY=(y+1)*stepY
        row.append((startX,startY,endX,endY))
        cell=warped[startY:endY, startX:endX]
        digit=extract_digit(cell,debug=args["debug"]>0)
        print(digit)

'''
   