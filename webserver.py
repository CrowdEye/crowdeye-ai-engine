from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ai import AiDetectionWorker
import threading
import time
from models import Node
from celery.task.control import revoke


import redis


webApi = Flask(__name__)
CORS(webApi)

# Main Index Stuff
@webApi.route("/", methods=["GET"], strict_slashes=False)
def index():
    return f"CrowdEye AI Detection Engine ACTIVE! Cameras Connected: {Node.get().all().count()}"


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

    if Node.get_by(nodeId=node_id):
        return f"Camera {node_id} Already Exists", 400

    print(f"Creating Node {node_id}")

    # Creating New Thread!!!
    # TODO: Add data to Redis
    # TODO: Start celery worker


    # newNode = NodeInfo(node_id, cam_ip)
    newNode = Node(nodeId=node_id, cameraIp=cam_ip)
    
    t_id = AiDetectionWorker.delay(newNode)
    newNode.thread = t_id
    return str(node_id)


# Remove Camera
@webApi.route("/remove_camera/<cam_id>", methods=["DELETE"], strict_slashes=False)
def remove_camera(cam_id):
    cam_id = str(cam_id)

    if not Node.get_by(nodeId=cam_id):
        return f"Camera {cam_id} Not Found", 400

    node = Node.get_by(nodeId=cam_id)

    # TODO: Remove cam from Redis
    # TODO: Kill celery worker
    revoke(node.thread, terminate=True)
    
    return "ok"


# Reset Camera
@webApi.route("/reset_camera/<cam_id>", methods=["POST"], strict_slashes=False)
def reset_camera(cam_id):
    cam_id = str(cam_id)

    if not Node.get_by(nodeId=cam_id):
        return f"Camera {cam_id} Not Found", 400
    node = Node.get_by(nodeId=cam_id)

    node.totalPeopleCount = 0
    node.totalLineCrossedLeft = 0
    node.totalLineCrossedRight = 0
    node.totalLineCrossed = 0
    
    return "ok"


# Get Cameras
@webApi.route("/get_cameras", methods=["GET"], strict_slashes=False)
def get_cameras():
    return jsonify(Node.query.all())


# Get Camera Info
@webApi.route("/camera/<cam_id>", methods=["GET"], strict_slashes=False)
def camera_info(cam_id):
    cam_id = str(cam_id)

    if not Node.get_by(nodeId=cam_id):
        return f"Camera {cam_id} Not Found", 400
    node = Node.get_by(nodeId=cam_id)

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

    if not Node.get_by(nodeId=cam_id):
        return f"Camera {cam_id} Not Found", 400
    node = Node.get_by(nodeId=cam_id)

    try:
        ax = int(data["ax"])
        ay = int(data["ay"])
        bx = int(data["bx"])
        by = int(data["by"])
    except KeyError:
        return "Missing Arguments", 400
    except ValueError:
        return "Arguments Have Wrong Datatype", 400

    node.lineAX = ax
    node.lineAY = ay
    node.lineBX = bx
    node.lineBY = by

    return "ok"


# Get Camera Raw Stream
@webApi.route("/stream/<cam_id>", methods=["GET"], strict_slashes=False)
def camera_stream(cam_id):
    cam_id = str(cam_id)

    if not Node.get_by(nodeId=cam_id):
        return f"Camera {cam_id} Not Found", 400
    node = Node.get_by(nodeId=cam_id)

    return Response(node.generateCameraStream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Get Camera Raw Stream
@webApi.route("/stream/<cam_id>/annotated", methods=["GET"], strict_slashes=False)
def camera_stream_annotated(cam_id):
    cam_id = str(cam_id)

    if not Node.get_by(nodeId=cam_id):
        return f"Camera {cam_id} Not Found", 400
    node = Node.get_by(nodeId=cam_id)

    return Response(node.generateAiStream(), mimetype='multipart/x-mixed-replace; boundary=frame')
