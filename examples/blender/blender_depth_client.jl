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

tup(point::Point3) = (point.x, point.y, point.z)

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

function open_mainfile!(client::BlenderClient, filepath)
    client.root[:open_mainfile](filepath)
    nothing
end

function setup_for_depth!(client::BlenderClient)
    client.root[:setup_for_depth]()
    nothing
end

function set_resolution!(client::BlenderClient, x, y)
    client.root[:set_resolution](x, y)
end

function add_plane!(client::BlenderClient, name::String, loc::Point3, rot::Point3, scale::Point3)
    client.root[:add_plane](name, tup(loc), tup(rot), tup(scale))
end

function set_object_location!(client::BlenderClient, name::String, point::Point3)
    client.root[:set_object_location](name, tup(point))
    nothing
end

function set_object_rotation_euler!(client::BlenderClient, name::String, point::Point3)
    client.root[:set_object_rotation_euler](name, tup(point))
    nothing
end

function set_object_scale!(client::BlenderClient, name::String, point::Point3)
    client.root[:set_object_scale](name, tup(point))
    nothing
end

function render(client::BlenderClient, filepath)
    client.root[:render](filepath)
end

client = BlenderClient()
connect!(client, "localhost", 59892)
setup_for_depth!(client)
set_resolution!(client, 100, 100)
set_object_location!(client, "Camera", Point3(0, -8.5, 5))
set_object_rotation_euler!(client, "Camera", Point3(pi/3., 0, 0))
add_plane!(client, "wall", Point3(0,10,0), Point3(pi/3.,0,0), Point3(20,20,20))
tic()
for i=1:100
    render(client, "$(pwd())/test_depth_$i.png")
end
toc()
