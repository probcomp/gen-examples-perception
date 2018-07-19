from body_pose_client import BodyPoseClient
import body_pose

client = BodyPoseClient(5003)
assert client.set_joint_rotation_euler([1, 2, 3]) == body_pose.OK
assert client.set_joint_rotation_euler([-1, -2, -3]) == body_pose.OK
