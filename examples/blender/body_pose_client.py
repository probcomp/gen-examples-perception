from json_socket_interface import JSONClient
import body_pose

class BodyPoseClient(object):
    
    def __init__(self, port):
        self.client = JSONClient(port)

    def set_joint_rotation_euler(self, euler_angles):
        return self.client.execute({
            "method" : body_pose.SET_JOINT_ROTATION_EULER,
            "data" : {
                "joint_name" : "foot",
                "euler_angles" : euler_angles
            }})

