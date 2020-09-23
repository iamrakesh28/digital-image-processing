import cv2
import numpy as np

THRESHOLD = 8
MAX = np.uint8(255)

class MotionSegment:

    # self.background -> background image
    # self.frame -> current frame
    # self.prevFrame -> previous frame
    # self.delay -> a parameter that sets the frame gap
    #               for constructing background image 
    def __init__(self, path):
        self.cam = cv2.VideoCapture(path)
        self.prevFrame = []
        self.frame = []
        self.background = []
        self.delay = 0

    # performs image subtraction
    # returns false on first frame
    # returns true on subsequent frames
    def segmentation(self, curFrame):

        # first frame (we don't display)
        if len(self.prevFrame) == 0:
            self.prevFrame = curFrame
            self.background = [[np.uint8(0) for j in range(len(curFrame[0]))]
                               for i in range(len(curFrame))]
            return False
        
        self.frame = []
        # after every two frames, it fills the empty spaces in the background
        self.delay = (self.delay + 1) % 2
        for row in range(len(curFrame)):
            self.frame.append([])
            #self.background.append([])
            for col in range(len(curFrame[row])):
                # if the intensities look similar then it is not the moving object
                # displays background as white
                if (abs(int(curFrame[row][col]) - int(self.prevFrame[row][col])) <= THRESHOLD):
                    self.frame[row].append(MAX)
                    # updates the background image
                    if self.background[row][col] == 0 and self.delay == 0:
                        self.background[row][col] = curFrame[row][col]
                else:
                    self.frame[row].append(self.prevFrame[row][col])
        self.prevFrame = curFrame
        self.frame = np.array(self.frame)
        return True

    # it was used for video cutting
    # not related with the problem (motionSegmentation)
    # only for the mario.avi file
    def saveVideo(self):
        success = True
        size = (int(self.cam.get(3)), int(self.cam.get(4)))
        out = cv2.VideoWriter('mario-new.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)
        countFrames = 0
        while success and countFrames < 130:
            success, frame = self.cam.read()
            #grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if countFrames > 70:
                out.write(frame)
            print(countFrames)
            #cv2.imshow('Moving Object', frame)
            #if cv2.waitKey(100) & 0xFF == ord('q'):
            #    break
            #print (grayFrame.shape)
            countFrames += 1
        self.cam.release()
        out.release()

    # performs motion segmentation
    def playVideo(self):

        countFrames = 0
        # reads the frame
        success, frame = self.cam.read()
        while success:
            # converts into grayscale image
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #print (grayFrame.shape, countFrames)
            #cv2.imshow('Moving Object', grayFrame)
            if self.segmentation(grayFrame):
                #print self.frame.shape
                # displays the moving object
                cv2.imshow('Moving Object', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            countFrames += 1
            success, frame = self.cam.read()

        # writes the bacground image
        cv2.imwrite('Background.jpg', np.array(self.background))
        self.cam.release()

if __name__ == "__main__":
    # reads the video file
    imageSeg = MotionSegment('mario-new.avi')
    #imageSeg.saveVideo()
    # motion segmentation
    imageSeg.playVideo()
