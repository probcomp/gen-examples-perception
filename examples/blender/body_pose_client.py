from json_socket_interface import JSONClient
import body_pose

class BodyPoseClient(object):
    
    def __init__(self, port):
        self.client = JSONClient(port)

    def _execute(self, method, data):
        return self.client.execute({
            "method" : method,
            "data" : data})

    def set_joint_rotation_euler(self, joint_name, euler_angles):
        return self._execute(body_pose.SET_JOINT_ROTATION_EULER, 
            {
                "joint_name" : joint_name,
                "euler_angles" : euler_angles
            })


    def capture_viewport(self, fname):
        return self._execute(body_pose.CAPTURE_VIEWPORT, { "fname" : fname })
