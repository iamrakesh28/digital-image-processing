import cv2
import numpy as np

THRESHOLD = 8
MAX = np.uint8(255)

class MotionSegment:
    def __init__(self, path):
        self.cam = cv2.VideoCapture(path)
        self.prevFrame = []
        self.frame = []
        self.background = []

    def segmentation(self, curFrame):
        
        if len(self.prevFrame) == 0:
            self.prevFrame = curFrame
            self.background = [[np.uint8(0) for j in range(len(curFrame[0]))]
                               for i in range(len(curFrame))]
            return False
        
        self.frame = []
        for row in range(len(curFrame)):
            self.frame.append([])
            #self.background.append([])
            for col in range(len(curFrame[row])):
                if (abs(int(curFrame[row][col]) - int(self.prevFrame[row][col])) <= THRESHOLD):
                    self.frame[row].append(MAX)
                    if self.background[row][col] == 0:
                        self.background[row][col] = curFrame[row][col]
                else:
                    self.frame[row].append(self.prevFrame[row][col])
        self.prevFrame = curFrame
        self.frame = np.array(self.frame)
        return True

    def playVideo(self):
        success = True
        while success:
            success, frame = self.cam.read()
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
           # print type(grayFrame)
            if self.segmentation(grayFrame):
                #print self.frame.shape
                cv2.imshow('Moving Object', self.frame)
            #print len(grayFrame), len(grayFrame[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.imwrite('Background.jpg', np.array(self.background))
        self.cam.release()

if __name__ == "__main__":
    imageSeg = MotionSegment(0)
    imageSeg.playVideo()
