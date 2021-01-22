import rom

class Node(rom.Model):
    nodeId = rom.String(required=True)
    cameraIp = rom.String(required=True)
    drawGui = rom.Boolean(default=True)
    renderToScreen = rom.Boolean(default=False)
    thread = rom.Integer()


    totalPeopleCount = rom.Integer(default=0)
    totalLineCrossedLeft = rom.Integer(default=0)
    totalLineCrossedRight = rom.Integer(default=0)
    totalLineCrossed = rom.Integer(default=0)

    lineAX = rom.Integer(default=318)
    lineAY = rom.Integer(default=0)
    lineBX = rom.Integer(default=318)
    lineBY = rom.Integer(default=637)
    cameraFrame = rom.String() # Technically binary but redis string are binary safe
    finishedFrame = rom.String() # Same, see: https://redis.io/topics/data-types
