from json_socket_interface import JSONClient
import body_pose

class BodyPoseClient(object):
    
    def __init__(self, port):
        self.client = JSONClient(port)

    def set_joint_rotation_euler(self, joint_name, euler_angles):
        return self.client.execute({
            "method" : body_pose.SET_JOINT_ROTATION_EULER,
            "data" : {
                "joint_name" : joint_name,
                "euler_angles" : euler_angles
            }})

