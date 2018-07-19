from body_pose_client import BodyPoseClient
import body_pose
import math
import time

client = BodyPoseClient(5003)
time.sleep(2)

assert client.set_joint_rotation_euler("arm_elbow_R", [0, 0, math.radians(0)]) == body_pose.OK
time.sleep(2)

assert client.set_joint_rotation_euler("arm_elbow_R", [0, 0, math.radians(45)]) == body_pose.OK
time.sleep(2)

assert client.set_joint_rotation_euler("arm_elbow_R", [0, 0, math.radians(90)]) == body_pose.OK
time.sleep(2)

assert client.set_joint_rotation_euler("arm_elbow_R", [0, 0, math.radians(135)]) == body_pose.OK
time.sleep(2)

assert client.set_joint_rotation_euler("arm_elbow_R", [0, 0, math.radians(180)]) == body_pose.OK
time.sleep(2)
