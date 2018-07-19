from body_pose_client import BodyPoseClient
import body_pose
import math
import time
#import os
#cwd = os.cwd()

start = time.time()
client = BodyPoseClient(5003)
for degrees in range(360):
    print(degrees)
    assert client.set_joint_rotation_euler("arm_elbow_R", [0, 0, math.radians(degrees)]) == body_pose.OK
    assert client.capture_viewport("test_%03d.png" % (degrees,)) == body_pose.OK
elapsed = time.time()
print(elapsed - start)
