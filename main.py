import sys
from webserver import *
from ai import *
import threading

print("Starting AI Engine Server...")
webApi.run(host='0.0.0.0', port="5500")
