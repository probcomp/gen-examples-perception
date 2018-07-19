import math
from json_socket_interface import JSONServer
#import bpy
import body_pose

class Pose(object):
    
    def __init__(self):
        #self.joints = bpy.data.objects['rig'].pose.bones
        self.joint_name_to_idx = {
            "arm_elbow_R" : 9,
            "arm_elbow_L" : 7,
            "hip" : 1,
            "heel_L" : 37,
            "heel_R" : 29
        }

    def get_joint(self, name):
        idx = self.joint_name_to_idx[name]
        return self.joints[idx]

    def set_joint_rotation_euler(self, **kwargs):
        #joint = self.get_joint(joint_name)
        #joint.rotation_euler[0] = euler_angles[0]
        #joint.rotation_euler[1] = euler_angles[1]
        #joint.rotation_euler[2] = euler_angles[2]
        return body_pose.OK

class BodyPoseServer(JSONServer):

    def __init__(self, port):
        super().__init__(port)
        self.pose = Pose()
        self.methods = {
            body_pose.SET_JOINT_ROTATION_EULER : self.pose.set_joint_rotation_euler
        }

    def process(self, request):
        method = self.methods[request["method"]]
        data = request["data"]
        return method(**data)

