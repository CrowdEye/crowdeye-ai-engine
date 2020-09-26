import sys
from webserver import *
from ai import *
import threading
import sys

print("Starting AI Engine Server...")
if (len(sys.argv) > 1):
    webApi.run(host='0.0.0.0', port="80")
else:
    webApi.run(host='0.0.0.0', port="5500")
