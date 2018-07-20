import math
import bpy
import os
import importlib.util
import threading

def import_module_from_cwd(module_name):
    cwd = os.getcwd()
    spec = importlib.util.spec_from_file_location(module_name, cwd + "/" + module_name + ".py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

body_pose = import_module_from_cwd("body_pose")
json_socket_interface = import_module_from_cwd("json_socket_interface")

class Pose(object):
    
    def __init__(self):
        self.joints = bpy.data.objects['rig'].pose.bones
        self.joint_name_to_idx = { # TODO add others?
            "arm_elbow_R" : 9,
            "arm_elbow_L" : 7,
            "hip" : 1,
            "heel_L" : 37,
            "heel_R" : 29
        }

    def get_joint(self, name):
        idx = self.joint_name_to_idx[name]
        return self.joints[idx]

    def set_joint_rotation_euler(self, joint_name, euler_angles):
        joint = self.get_joint(joint_name)
        joint.rotation_mode = 'XYZ'
        joint.rotation_euler[0] = euler_angles[0]
        joint.rotation_euler[1] = euler_angles[1]
        joint.rotation_euler[2] = euler_angles[2]
        return body_pose.OK

    def capture_viewport(self, fname):
        print("capture_viewport with fname: ", fname)
        #bpy.data.scenes["Scene"].use_nodes = False    
        bpy.context.scene.render.filepath = fname
        bpy.ops.render.opengl(write_still=True)
        return body_pose.OK

    def capture_viewport_depth(self, fname):
        bpy.context.scene.render.layers['RenderLayer'].use_pass_mist = True
        scene.world.mist_settings.falloff = 'LINEAR'
        scene.world.mist_settings.intensity = 0.0 # minimum depth
        scene.world.mist_settings.depth = dist # maximum depth
        bpy.context.scene.render.filepath = fname
        bpy.ops.render.opengl(write_still=True)
        return body_pose.OK


class BodyPoseServer(json_socket_interface.JSONServer):

    def __init__(self, port):
        super().__init__(port)
        self.pose = Pose()
        self.methods = {
            body_pose.SET_JOINT_ROTATION_EULER : self.pose.set_joint_rotation_euler,
            body_pose.CAPTURE_VIEWPORT : self.pose.capture_viewport
        }

    def process(self, request):
        method = self.methods[request["method"]]
        data = request["data"]
        return method(**data)

print("HI!")
server = BodyPoseServer(5003)
server.run()
#thread = threading.Thread(target=server.run, args=())
#thread.daemon = True
#thread.start()
