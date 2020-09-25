import sys
from webserver import *
from ai import *
import threading

'''
parameters = {}
parameters["drawGui"] = True
parameters["renderToScreen"] = True
parameters["showAll"] = False
parameters["cameraId"] = 0
parameters["cameraIp"] = "http://localhost:8080/stream.mjpg"
 
AIThread = threading.Thread(target=AiDetectionWorker, args=(parameters,))
AIThread.setDaemon(True)

AIThread.start()
AIThread.join()
'''

testNode = NodeInfo(127, "http://localhost:8080/stream.mjpg", renderToScreen=True)
 
AIThread = threading.Thread(target=AiDetectionWorker, args=(testNode,))
AIThread.setDaemon(True)

AIThread.start()
AIThread.join()


#AiDetectionWorker(parameters)

