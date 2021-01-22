import sys
import os
from webserver import *
from ai import *
import threading
import sys

from rom import util
util.set_connection_settings(host='redis', db=0)


print("Starting AI Engine Server...")
if (len(sys.argv) > 1):
    webApi.run(host='0.0.0.0', port=os.environ.get("PORT", 80))
else:
    webApi.run(host='0.0.0.0', port="5500")
