using PyCall

@pyimport rpyc

############
# 3D point #
############

struct Point3
    x::Float64
    y::Float64
    z::Float64
end

Point3(tup::Tuple{U,U,U}) where {U<:Real} = Point3(tup[1], tup[2], tup[3])

Base.:+(a::Point3, b::Point3) = Point3(a.x + b.x, a.y + b.y, a.z + b.z)

tup(point::Point3) = (point.x, point.y, point.z)


#############
# body pose #
#############

struct BodyPose
    rotation::Point3
    arm_elbow_r_location::Point3
    arm_elbow_l_location::Point3
    arm_elbow_r_rotation::Point3
    arm_elbow_l_rotation::Point3
    hip_location::Point3
    heel_r_location::Point3
    heel_l_location::Point3
end

function Base.:+(a::BodyPose, b::BodyPose)
    BodyPose(
        a.rotation + b.rotation,
        a.arm_elbow_r_location + b.arm_elbow_r_location,
        a.arm_elbow_l_location + b.arm_elbow_l_location,
        a.arm_elbow_r_rotation + b.arm_elbow_r_rotation,
        a.arm_elbow_l_rotation + b.arm_elbow_l_rotation,
        a.hip_location + b.hip_location,
        a.heel_r_location + b.heel_r_location,
        a.heel_l_location + b.heel_l_location)
end

const RIG = "rig"
const ARM_ELBOW_R = "arm elbow_R"
const ARM_ELBOW_L = "arm elbow_L"
const HIP = "hip"
const HEEL_R = "heel_R"
const HEEL_L = "heel_L"

##################
# blender client #
##################

mutable struct BlenderClient
    conn::PyObject
    root::PyObject
    BlenderClient() = new()
end

function connect!(client::BlenderClient, host, port)
    client.conn = rpyc.connect(host, port)
    client.root = client.conn[:root]
end

function setup_for_depth!(client::BlenderClient)
    client.root[:setup_for_depth]()
    nothing
end

function setup_for_wireframe!(client::BlenderClient)
    client.root[:setup_for_wireframe]()
    nothing
end

function set_resolution!(client::BlenderClient, x, y)
    client.root[:set_resolution](x, y)
end

function add_plane!(client::BlenderClient, object::String, loc::Point3, rot::Point3, scale::Point3)
    client.root[:add_plane](object, tup(loc), tup(rot), tup(scale))
end

function set_object_location!(client::BlenderClient, object::String, point::Point3)
    client.root[:set_object_location](object, tup(point))
    nothing
end

function set_object_rotation_euler!(client::BlenderClient, object::String, point::Point3)
    client.root[:set_object_rotation_euler](object, tup(point))
    nothing
end

function set_object_scale!(client::BlenderClient, object::String, point::Point3)
    client.root[:set_object_scale](object, tup(point))
    nothing
end

function get_object_location!(client::BlenderClient, object::String)
    Point3(client.root[:get_object_location](object))
end

function get_object_rotation_euler(client::BlenderClient, object::String)
    Point3(client.root[:get_object_rotation_euler](object))
end

function get_object_scale(client::BlenderClient, object::String)
    Point3(client.root[:get_object_scale](object))
end

function set_bone_location(client::BlenderClient, object::String, bone::String, location::Point3)
    client.root[:set_bone_location](object, bone, tup(location))
    nothing
end

function set_bone_rotation_euler!(client::BlenderClient, object::String, bone::String, rotation_euler::Point3)
    client.root[:set_bone_rotation_euler](object, bone, tup(rotation_euler))
    nothing
end

function get_bone_location(client::BlenderClient, object::String, bone::String)
    Point3(client.root[:get_bone_location](object, bone))
end

function get_bone_rotation_euler(client::BlenderClient, object::String, bone::String)
    Point3(client.root[:get_bone_rotation_euler](object, bone))
end

function render(client::BlenderClient, filepath)
    client.root[:render](filepath)
end

function get_body_pose(client::BlenderClient)
    BodyPose(
        get_object_rotation_euler(client, RIG),
        get_bone_location(client, RIG, ARM_ELBOW_R),
        get_bone_location(client, RIG, ARM_ELBOW_L),
        get_bone_rotation_euler(client, RIG, ARM_ELBOW_R),
        get_bone_rotation_euler(client, RIG, ARM_ELBOW_L),
        get_bone_location(client, RIG, HIP),
        get_bone_location(client, RIG, HEEL_R),
        get_bone_location(client, RIG, HEEL_L))
end

function set_body_pose!(client::BlenderClient, pose::BodyPose)
    pose_dict = Dict(
        "rotation" => tup(pose.rotation),
        "arm_elbow_r_location" => tup(pose.arm_elbow_r_location),
        "arm_elbow_l_location" => tup(pose.arm_elbow_l_location),
        "arm_elbow_r_rotation" => tup(pose.arm_elbow_r_rotation),
        "arm_elbow_l_rotation" => tup(pose.arm_elbow_l_rotation),
        "hip_location" => tup(pose.hip_location),
        "heel_r_location" => tup(pose.heel_r_location),
        "heel_l_location" => tup(pose.heel_l_location))
    client.root[:set_body_pose](pose_dict)
end
