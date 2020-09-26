from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ai import NodeInfo, AiDetectionWorker
import threading
import time
webApi = Flask(__name__)
CORS(webApi)

# Dict Of All Cameras + Ids
cameras = {}


# Main Index Stuff
@webApi.route("/", methods=["GET"], strict_slashes=False)
def index():
    return f"CrowdEye AI Detection Engine ACTIVE! Cameras Connected: {len(cameras.keys())}"


# Add Camera
@webApi.route("/add_camera", methods=["POST"], strict_slashes=False)
def add_camera():
    data = request.get_json(force=True)
    try:
        cam_ip = str(data["cam_ip"])
        node_id = str(data["node_id"])
    except KeyError:
        return "Missing Arguments", 400
    except ValueError:
        return "Arguments Have Wrong Datatype", 400

    if node_id in cameras:
        return f"Camera {node_id} Already Exists", 400

    print(f"Creating Node {node_id}")

    # Creating New Thread!!!
    newNode = NodeInfo(node_id, cam_ip)
    nodeThread = threading.Thread(target=AiDetectionWorker, args=(newNode,))
    nodeThread.setDaemon(True)
    newNode.thread = newNode

    # Start Thread
    nodeThread.start()

    # Add To Camera Dict
    cameras[node_id] = newNode
    
    return str(node_id)


# Remove Camera
@webApi.route("/remove_camera/<cam_id>", methods=["GET"], strict_slashes=False)
def remove_camera(cam_id):
    cam_id = str(cam_id)

    if cam_id not in cameras:
        return f"Camera {cam_id} Not Found", 400
    node = cameras[cam_id]

    # Set Stop Flag
    node.active = False

    # Wait for server to stop
    print(f"Waiting To Stop Server {cam_id}")
    while True:
        time.sleep(0.5)
        if node.active is None:
            break

    del cameras[cam_id]
    
    return "ok"


# Get Cameras
@webApi.route("/get_cameras", methods=["GET"], strict_slashes=False)
def get_cameras():
    return jsonify(list(cameras.keys()))


# Get Camera Info
@webApi.route("/camera/<cam_id>", methods=["GET"], strict_slashes=False)
def camera_info(cam_id):
    cam_id = str(cam_id)

    if cam_id not in cameras:
        return f"Camera {cam_id} Not Found", 400
    node = cameras[cam_id]

    response = {}
    response["node_id"] = node.nodeId
    response["camera_ip"] = node.cameraIp
    response["total_people"] = node.totalPeopleCount
    response["crossed_left"] = node.totalLineCrossedLeft
    response["crossed_right"] = node.totalLineCrossedRight
    response["total_crossed"] = node.totalLineCrossed

    return jsonify(response)


# Change Camera Line Info
@webApi.route("/change_line/<cam_id>", methods=["POST"], strict_slashes=False)
def camera_change_line(cam_id):
    data = request.get_json(force=True)
    cam_id = str(cam_id)

    if cam_id not in cameras:
        return f"Camera {cam_id} Not Found", 400
    node = cameras[cam_id]

    try:
        ax = int(data["ax"])
        ay = int(data["ay"])
        bx = int(data["bx"])
        by = int(data["by"])
    except KeyError:
        return "Missing Arguments", 400
    except ValueError:
        return "Arguments Have Wrong Datatype", 400

    node.lineA = (ax, ay)
    node.lineB = (bx, by)

    return "ok"


# Get Camera Raw Stream
@webApi.route("/stream/<cam_id>", methods=["GET"], strict_slashes=False)
def camera_stream(cam_id):
    cam_id = str(cam_id)

    if cam_id not in cameras:
        return f"Camera {cam_id} Not Found", 400
    node = cameras[cam_id]

    return Response(node.generateCameraStream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Get Camera Raw Stream
@webApi.route("/stream/<cam_id>/annotated", methods=["GET"], strict_slashes=False)
def camera_stream_annotated(cam_id):
    cam_id = str(cam_id)

    if cam_id not in cameras:
        return f"Camera {cam_id} Not Found", 400
    node = cameras[cam_id]

    return Response(node.generateAiStream(), mimetype='multipart/x-mixed-replace; boundary=frame')
