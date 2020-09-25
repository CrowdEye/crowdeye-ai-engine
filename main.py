import sys
from ai import *
import threading

parameters = {}

parameters["drawGui"] = True
parameters["renderToScreen"] = True
parameters["showAll"] = False
parameters["cameraId"] = 0
parameters["cameraIp"] = "http://localhost:8080/video.mjpg"
 
AIThread = threading.Thread(target=AiDetectionWorker, args=(parameters,))
AIThread.setDaemon(True)

AIThread.start()
AIThread.join()


#AiDetectionWorker(parameters)