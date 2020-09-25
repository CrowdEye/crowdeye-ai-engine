import sys
from webserver import *
from ai import *
import threading

'''
print("Starting AI Engine Server...")
webApi.run(host='0.0.0.0', port="5500")
'''

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

testNode = NodeInfo(127, "http://localhost:5510/stream.mjpg", renderToScreen=True)
 
AiDetectionWorker(testNode)


#AiDetectionWorker(parameters)

