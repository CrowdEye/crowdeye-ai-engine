# Hack thing to remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Imports
from utils.datasets import *
from utils.models import *
from utils.sort import *
from utils.utils import *
from torchvision import transforms
from torch.autograd import Variable
from collections import deque
import urllib.request
import threading
import numpy
import torch
import cv2
import math
import time
import os

# load model and put into eval mode
imgSize = 416
model = Darknet("model/yolov3.cfg", img_size=imgSize)
model.load_weights("model/yolov3.weights")
model.cuda()
model.eval()

# Load Classes
fp = open("model/yolov3.classes", "r")
classes = fp.read().split("\n")[:-1]
# print(classes)

defaultFont = cv2.FONT_HERSHEY_SIMPLEX

# Frame Stuff
currentFrame = None
finishedFrame = None

# Line Infos
lineA = (318,0)
lineB = (318,637)

colours=[
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,0,255),
    (128,0,0),
    (0,128,0),
    (0,0,128),
    (128,0,128),
    (128,128,0),
    (0,128,128)
]

def openIpCam(ip):
    global currentFrame
    with urllib.request.urlopen(ip) as url:
        inBytes = bytes()
        while True:
            inBytes += url.read(1024)
            a = inBytes.find(b'\xff\xd8')
            b = inBytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = inBytes[a:b+2]
                inBytes = inBytes[b+2:]
                i = cv2.imdecode(numpy.fromstring(jpg, dtype=numpy.uint8), cv2.IMREAD_COLOR)
                currentFrame = i

def runDetection(img):
    # Scale + Pad Image
    # Code Stolen From Stack
    ratio = min(imgSize/img.size[0], imgSize/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])

    # Convert Image
    imageTensor = img_transforms(img).float()
    imageTensor = imageTensor.unsqueeze_(0)
    inputImg = Variable(imageTensor.type(torch.cuda.FloatTensor))

    # Get Detections
    with torch.no_grad():
        detections = model(inputImg)
        detections = non_max_suppression(detections, 80, 0.8, 0.4)
        
    #print(detections)
    #print(detections[0])
    return detections[0]

def AiDetectionWorker(parameters):
    # Get Params
    #print(parameters)
    drawGui = parameters["drawGui"]
    renderToScreen = parameters["renderToScreen"]
    cameraId = parameters["cameraId"]
    cameraIp = parameters["cameraIp"]

    print(f"Creating AI Worker For Camera {cameraId}")

    # Create Motion Tracker (Sort is the ALGO)
    motionTracker = Sort() 

    pointsDict = {}
    uniqueIdList = []
    LineCrossingIdCache = [] # List of ids which are currantly crossing the line

    # Test Params
    totalPeopleCount = 0
    totalLineCrossedLeft = 0
    totalLineCrossedRight = 0
    totalLineCrossed = 0

    # Start Ip Camera
    IPCamThread = threading.Thread(target=openIpCam, args=(cameraIp,))
    IPCamThread.setDaemon(True)
    IPCamThread.start()

    # Wait For Connection
    print("Connecting To Ip Camera.", end="")
    while(True):
        if currentFrame is None:
            print(".", end="")
            time.sleep(0.5)
        else:
            print("")
            break

    # Start Detection Loop
    frames = 0
    print(f"Starting AI Loop")
    startTime = time.time()
    while(True):
        frames += 1
        # print(frames)

        # People
        peoplecount = 0
        totalSpeed = 0

        # Render and Parse Image Into DarkNet
        frame = currentFrame
        pilImg = Image.fromarray(cv2.cvtColor(currentFrame, cv2.COLOR_RGB2BGR))
        detections = runDetection(pilImg)
        img = numpy.array(pilImg)

        # Get Padding
        padX = max(img.shape[0]-img.shape[1], 0)*(imgSize/max(img.shape))
        padY = max(img.shape[1]-img.shape[0], 0)*(imgSize/max(img.shape))
        upadX = imgSize - padX
        upadY = imgSize - padY

        #print(padX, padY, upadX, upadY)

        # Get Detections
        if detections is not None:
            trackedObjects = motionTracker.update(detections.cpu())
        else:
            trackedObjects = []

        # print("[", end="")

        # For Every Detection, Run This
        for x1, y1, x2, y2, objId, objIndex in trackedObjects:
            detectedObj = classes[int(objIndex)]
            # print(detectedObj)
            # print(detectedObj + " ", end="")

            # Check If Detected Object Is A Person
            if detectedObj == "person":
                # Turn Id Into Int
                objId = int(objId)

                # Speed Stuff
                speed = 0
                xdir = ""
                ydir = ""

                # Generate Bounding Boxes
                boxY1 = int(((y1-padY//2)/upadY)*img.shape[0])
                boxX1 = int(((x1-padX//2)/upadX)*img.shape[1])
                boxH = int(((y2-y1)/upadY)*img.shape[0])
                boxW = int(((x2-x1)/upadX)*img.shape[1])
                # Get Center
                center = (round(boxX1 + (boxW / 2)), round(boxY1 + (boxH / 2)))
                # print(center)
                # print(uniqueIdList)
                # print(lineA, lineB)

                # Count People
                peoplecount += 1 
                if objId not in uniqueIdList:
                    # print(objId)
                    print(f"Detected New Person {objId}")
                    uniqueIdList.append(objId)

                # Add Dot To Tracking List
                if objId in pointsDict:
                    pointsDict[objId].appendleft(center)
                else:
                    pointsDict[objId] = deque(maxlen=26)
                    pointsDict[objId].appendleft(center)

                trackingPointsList = pointsDict[objId]

                #print(pointsDict)
                    
                # Generate Diff In X and Y
                dx = 0
                dy = 0
                for x in range(len(trackingPointsList)-1):
                    cv2.line(frame, trackingPointsList[x], trackingPointsList[x+1], (0, 255, 0), 10)
                    dx += trackingPointsList[x+1][0] - trackingPointsList[x][0]  
                    dy += trackingPointsList[x+1][1] - trackingPointsList[x][1] 
                # print(Id, dx, dy)

                # Start Checking Line If People
                if len(pointsDict[objId]) > 6:
                    x = ""
                    y = ""
                    if(dx < 0):
                        xdir = "right"
                    if(dx > 0):
                        xdir = "left"
                    if(dy < 0):
                        ydir = "down"
                    if(dy > 0):
                        ydir = "up"

                    # Calculate Speed
                    speed = round(math.sqrt(abs(dx*dx-dy*dy))/10)

                    # Every 10 Frames, Calculate Line Cross
                    if(frames % 10 == 0):
                        lineCrossed = getCountLineCrossed(pointsDict[objId])
                        if lineCrossed != None:
                            if objId not in LineCrossingIdCache:
                                if lineCrossed == "left":
                                    print("Detection - Cross Left")
                                    totalLineCrossedLeft += 1
                                elif lineCrossed == "right":
                                    print("Detection - Cross Right")
                                    totalLineCrossedRight += 1
                                totalLineCrossed += 1
                                LineCrossingIdCache.append(objId)
                        else:
                            if objId in LineCrossingIdCache:
                                LineCrossingIdCache.remove(objId)

                # Draw GUI Stuff
                if drawGui:
                    colour = colours[objId % len(colours)]
                    cv2.rectangle(frame, (boxX1, boxY1), (boxX1+boxW, boxY1+boxH), colour, 4)
                    cv2.rectangle(frame, (boxX1, boxY1-105), (boxX1+len(detectedObj)*19+80, boxY1), colour, -1)
                    cv2.putText(frame, f"ID: " + str(objId), (boxX1, boxY1 - 10), defaultFont, 1, (255,255,255), 3)
                    cv2.putText(frame, f"{xdir} - {ydir}", (boxX1, boxY1 - 35), defaultFont, 1, (255,255,255), 3)
                    cv2.putText(frame, f"Speed: {speed}", (boxX1, boxY1 - 70), defaultFont, 1, (255,255,255), 3)
                    cv2.circle(frame, center, 5, (0, 0, 255), 5)

        # print("]")

        # Draw GUI Stuff
        if drawGui:
            cv2.line(frame, lineA, lineB, (255, 0, 255), 10)
            cv2.rectangle(frame, (0, 15), (235, 35), (0,0,0), -1)
            cv2.rectangle(frame, (0, 450), (200, 500), (0,0,0), -1)
            cv2.putText(frame, f"=CrowdEye= Camera Node {cameraId}", (0, 30), defaultFont, 0.5, (0,165,255), 2)     
            cv2.putText(frame, f"People Detected: {peoplecount}", (0, 60), defaultFont, 0.5, (255,255,0), 2)     
            cv2.putText(frame, f"Total People Count: {len(uniqueIdList)}", (0, 90), defaultFont, 0.5, (255,255,0), 2)
            cv2.putText(frame, f"People Count Cross Line Left: {totalLineCrossedLeft}", (0, 120), defaultFont, 0.5, (255,255,0), 2)            
            cv2.putText(frame, f"People Count Cross Line Right: {totalLineCrossedRight}", (0, 150), defaultFont, 0.5, (255,255,0), 2)            
            cv2.putText(frame, f"People Count Cross Line Total: {totalLineCrossed}", (0, 180), defaultFont, 0.5, (255,255,0), 2)           
            cv2.putText(frame, f"Video Detection FPS: {round(frames / (time.time() - startTime))}", (0, 470), defaultFont, 0.5, (255,255,255), 2)

        # Render Frame To Variable
        finishedFrame = frame
        # Check If Should Render To Secreen    
        if(renderToScreen):
            cv2.imshow(f"CrowdEye Camera Node {cameraId}", frame)
            key = 0xFF & cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                print("ENDING DETECTION")
                break


# Line Corssing Algo (stolen online)
def getCountLineCrossed(trackingPointsList):
    position = ((lineB[0] - lineA[0])*(trackingPointsList[0][1] - lineA[1]) - (lineB[1] - lineA[1])*(trackingPointsList[0][0] - lineA[0]))
    prevposition = ((lineB[0] - lineA[0])*(trackingPointsList[0 - 5][1] - lineA[1]) - (lineB[1] - lineA[1])*(trackingPointsList[0 - 5][0] - lineA[0]))
    #print(position, prevposition)
    if(prevposition != 0 and position != 0):
        if(position > 0 and prevposition < 0):
            return "right"
        if(position < 0 and prevposition > 0):
            return "left"
    return None

# Close Windows
cv2.destroyAllWindows()
