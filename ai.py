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
from turbojpeg import TurboJPEG
import torch
from collections import deque
import urllib.request
import threading
import numpy
import torch
import cv2
import math
import time
import os
import sys

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

# Frame Stuff
screenX, screenY = (640, 480)
defaultFont = cv2.FONT_HERSHEY_SIMPLEX
connectingFrame = np.zeros(shape=[screenY, screenX, 3], dtype=np.uint8)
cv2.putText(connectingFrame, "[CrowdEye] Connecting To IP Camera...", (0, 30), defaultFont, 0.5, (255, 255, 255), 2)
if (len(sys.argv) > 1):
    turbojpeg = TurboJPEG("/usr/lib/x86_64-linux-gnu/libturbojpeg.so")

# COLOURS!!!
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


# Define Config Class
class NodeInfo:
    def __init__(self, nodeId, cameraIp, drawGui=True, renderToScreen=False):
        # Important Node Params
        self.nodeId = nodeId
        self.cameraIp = cameraIp
        self.drawGui = drawGui
        self.renderToScreen = renderToScreen
        self.thread = None
        self.active = True  # Hacky Way To Remotely Stop Thread

        # Tracker Information
        self.totalPeopleCount = 0
        self.totalLineCrossedLeft = 0
        self.totalLineCrossedRight = 0
        self.totalLineCrossed = 0

        # Line Info
        self.lineA = (318,0)
        self.lineB = (318,637)

        # Video Frames
        self.cameraFrame = None
        self.finishedFrame = None

    def generateCameraStream(self):
        print(f"[NODE {self.nodeId}] Generating Camera Stream")
        while True:
            time.sleep(0.02)
            if self.cameraFrame is not None:
                img = self.cameraFrame
            else:
                img = connectingFrame
            if (len(sys.argv) > 1):
                jpg = turbojpeg.encode(img, quality=40)
                frame = jpg
            else:
                _, jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                frame = jpg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def generateAiStream(self):
        print(f"[NODE {self.nodeId}] Generating AI Stream")
        while True:
            time.sleep(0.02)
            if self.finishedFrame is not None:
                img = self.finishedFrame
            else:
                img = connectingFrame
            if (len(sys.argv) > 1):
                jpg = turbojpeg.encode(img, quality=40)
            else:
                _, jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                frame = jpg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Open Connection To Ip Camera
def openIpCam(nodeInfo):
    while True:
        if nodeInfo.active != True:
            break  # Node Stop
        try:
            with urllib.request.urlopen(nodeInfo.cameraIp) as url:
                inBytes = bytes()
                while True:
                    if nodeInfo.active != True:
                        nodeInfo.cameraFrame = None
                        break  # Node Stop
                    time.sleep(0.01)
                    inBytes += url.read1()
                    a = inBytes.find(b'\xff\xd8')
                    b = inBytes.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = inBytes[a:b+2]
                        inBytes = inBytes[b+2:]
                        if (len(sys.argv) > 1):
                            i = turbojpeg.decode(jpg)
                        else:
                            i = cv2.imdecode(numpy.fromstring(jpg, dtype=numpy.uint8), cv2.IMREAD_COLOR)
                        nodeInfo.cameraFrame = i
        except Exception as e:
            print(f"[NODE {nodeInfo.nodeId}] Open Camera Error: {e}")
            nodeInfo.cameraFrame = None
        time.sleep(0.5)


# Actual Detection Stuff
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
    imageTensor = imageTensor.unsqueeze_(0).to("cuda:0")
    inputImg = Variable(imageTensor.type(torch.cuda.FloatTensor))

    # Get Detections
    with torch.no_grad():
        detections = model(inputImg)
        detections = non_max_suppression(detections, 80, 0.8, 0.4)
        
    #print(detections)
    #print(detections[0])
    return detections[0]


# Ai Worker Thread
def AiDetectionWorker(nodeInfo):
    # Get Params
    #print(parameters)
    #print(nodeInfo)
    print(f"[NODE {nodeInfo.nodeId}] Creating AI Worker For Camera {nodeInfo.nodeId}")

    # Create Motion Tracker (Sort is the ALGO)
    motionTracker = Sort() 

    pointsDict = {}
    uniqueIdList = []
    LineCrossingIdCache = [] # List of ids which are currantly crossing the line

    # Start Ip Camera
    IPCamThread = threading.Thread(target=openIpCam, args=(nodeInfo,))
    IPCamThread.setDaemon(True)
    IPCamThread.start()

    # Wait For Connection
    '''
    print("Connecting To Ip Camera.", end="")
    while(True):
        if cameraFrame is None:
            print(".", end="")
            time.sleep(0.5)
        else:
            print("")
            break
    '''

    # Start Detection Loop
    frames = 0
    print(f"[NODE {nodeInfo.nodeId}] Starting AI Loop")
    startTime = None
    while(True):
        time.sleep(0.01)
        if(nodeInfo.active == False):
            print(f"[NODE {nodeInfo.nodeId}] Stopping Ai Loop")
            nodeInfo.finishedFrame = None
            nodeInfo.cameraFrame = None
            # Set To None to Signal Successful Closure!!
            nodeInfo.active = None
            # Break Out Of Loop And Stop Server
            break
        # Print Custom Thing If No Camera
        if nodeInfo.cameraFrame is not None:
            if startTime is None:
                # Hacky thing to get accurate fps counter even when cam is down
                frames = 0
                startTime = time.time()
            frames += 1
            peoplecount = 0
            # print(frames)

            # Render and Parse Image Into DarkNet
            frame = nodeInfo.cameraFrame
            frame = cv2.resize(frame, (screenX, screenY))
            pilImg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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
                        print(f"[NODE {nodeInfo.nodeId}] Detected New Person {objId}")
                        uniqueIdList.append(objId)
                        nodeInfo.totalPeopleCount = len(uniqueIdList)

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
                        speed = round(math.sqrt(abs(dx*dx-dy*dy))/25)

                        # Every 10 Frames, Calculate Line Cross
                        if(frames % 10 == 0):
                            lineCrossed = getCountLineCrossed(nodeInfo.lineA, nodeInfo.lineB, pointsDict[objId])
                            if lineCrossed != None:
                                if objId not in LineCrossingIdCache:
                                    if lineCrossed == "left":
                                        print(f"[NODE {nodeInfo.nodeId}] Detection - Cross Left")
                                        nodeInfo.totalLineCrossedLeft += 1
                                    elif lineCrossed == "right":
                                        print(f"[NODE {nodeInfo.nodeId}] Detection - Cross Right")
                                        nodeInfo.totalLineCrossedRight += 1
                                    nodeInfo.totalLineCrossed += 1
                                    LineCrossingIdCache.append(objId)
                            else:
                                if objId in LineCrossingIdCache:
                                    LineCrossingIdCache.remove(objId)

                    # Draw GUI Stuff
                    if nodeInfo.drawGui:
                        colour = colours[objId % len(colours)]
                        cv2.rectangle(frame, (boxX1, boxY1), (boxX1+boxW, boxY1+boxH), colour, 4)
                        cv2.rectangle(frame, (boxX1, boxY1-105), (boxX1+len(detectedObj)*19+80, boxY1), colour, -1)
                        cv2.putText(frame, f"ID: {objId}", (boxX1, boxY1 - 10), defaultFont, 1, (255,255,255), 3)
                        cv2.putText(frame, f"{xdir} - {ydir}", (boxX1, boxY1 - 35), defaultFont, 1, (255,255,255), 3)
                        cv2.putText(frame, f"Speed: {speed}", (boxX1, boxY1 - 70), defaultFont, 1, (255,255,255), 3)
                        cv2.circle(frame, center, 5, (0, 0, 255), 5)

            # print("]")

            # Draw GUI Stuff
            if nodeInfo.drawGui:
                cv2.line(frame, nodeInfo.lineA, nodeInfo.lineB, (0,255,0), 10)
                cv2.rectangle(frame, (0, 15), (235, 35), (0,0,0), -1)
                cv2.rectangle(frame, (0, 450), (200, 500), (0,0,0), -1)
                cv2.putText(frame, f"=CrowdEye= Camera Node {nodeInfo.nodeId}", (0, 30), defaultFont, 0.5, (0,165,255), 2)     
                cv2.putText(frame, f"People Detected: {peoplecount}", (0, 60), defaultFont, 0.5, (255, 255, 255), 2)     
                cv2.putText(frame, f"Total People Count: {len(uniqueIdList)}", (0, 90), defaultFont, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"People Count Cross Line Left: {nodeInfo.totalLineCrossedLeft}", (0, 120), defaultFont, 0.5, (255, 255, 255), 2)            
                cv2.putText(frame, f"People Count Cross Line Right: {nodeInfo.totalLineCrossedRight}", (0, 150), defaultFont, 0.5, (255, 255, 255), 2)            
                cv2.putText(frame, f"People Count Cross Line Total: {nodeInfo.totalLineCrossed}", (0, 180), defaultFont, 0.5, (255, 255, 255), 2)           
                cv2.putText(frame, f"Video Detection FPS: {round(frames / (time.time() - startTime))}", (0, screenY-10), defaultFont, 0.5, (255,255,255), 2)
        else:
            startTime = None
            frame = connectingFrame  

        # Render Frame To Variable
        nodeInfo.finishedFrame = frame 
        # Check If Should Render To Secreen    
        if(nodeInfo.renderToScreen):
            cv2.imshow(f"CrowdEye Camera Node {nodeInfo.nodeId}", frame)
            key = 0xFF & cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                print(f"[NODE {nodeInfo.nodeId}] ENDING DETECTION")
                break


# Line Corssing Algo (stolen online)
def getCountLineCrossed(lineA, lineB, trackingPointsList):
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
# cv2.destroyAllWindows()
