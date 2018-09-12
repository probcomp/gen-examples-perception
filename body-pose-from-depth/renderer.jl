using FileIO
using ImageFiltering: imfilter, Kernel

include("blender_depth_client.jl")

############
# renderer #
############

struct BodyPoseRenderer
    width::Int
    height::Int
    blender_client::BlenderClient
end

function BodyPoseRenderer(width, height, hostname, port)
    client = BlenderClient()
    connect!(client, hostname, port)
    setup_for_depth!(client)
    set_resolution!(client, width, height)
    set_object_location!(client, "Camera", Point3(0, -8.5, 5))
    set_object_rotation_euler!(client, "Camera", Point3(pi/3., 0, 0))
    add_plane!(client, "background", Point3(0,4,0), Point3(pi/3.,0,0), Point3(20,20,20))
    set_object_location!(client, RIG, Point3(0, 0, 0))
    set_object_rotation_euler!(client, RIG, Point3(0, 0, 0))
    set_object_scale!(client, RIG, Point3(3, 3, 3))
    add_plane!(client, "nearplane", Point3(-2,-4,0), Point3(pi/3.,0,0), Point3(0.1,0.1,0.1))
    setup_for_depth!(client)
    set_resolution!(client, width, height)
    return BodyPoseRenderer(width, height, client)
end

function render(renderer::BodyPoseRenderer, pose::BodyPose)
    tmp = tempname() * ".png"
    set_body_pose!(renderer.blender_client, pose)
    render(renderer.blender_client, tmp)
    img = FileIO.load(tmp)
    rm(tmp)
    return convert(Matrix{Float64}, img)
end

