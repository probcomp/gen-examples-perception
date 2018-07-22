using PyCall

@pyimport rpyc

struct Wall
    loc_x::Float64
    loc_y::Float64
    loc_z::Float64
    rotation_euler_1::Float64
    rotation_euler_2::Float64
    rotation_euler_3::Float64
    scale_x::Float64
    scale_y::Float64
    scale_z::Float64
end

############
# 3D point #
############

struct Point3
    x::Float64
    y::Float64
    z::Float64
end

function tup(point::Point3)
    (point.x, point.y, point.z)
end

##################
# blender object #
##################

struct Object
    obj::PyObject
end

function set_location!(object::Object, point::Point3)
    object.obj[:location] = tup(point)
    nothing
end

function set_rotation_euler!(object::Object, euler::Point3)
    object.obj[:rotation_euler] = tup(euler)
    nothing
end

function set_scale!(object::Object, scale::Point3)
    object.obj[:scale] = tup(scale)
    nothing
end

function set_hide_render!(object::Object, hide::Bool)
    object.obj[:hide_render] = hide
    nothing
end

##################
# blender client #
##################

mutable struct BlenderClient
    bpy::PyObject
    scene::PyObject
    objects::Dict{String,Object}
    BlenderClient() = new()
end

function set_internal_render_engine!(client::BlenderClient)
    client.scene[:render][:engine] = "BLENDER_RENDER"
    nothing
end

function set_cycles_render_engine!(client::BlenderClient)
    client.scene[:render][:engine] = "CYCLES"
    nothing
end

function get_by_name(bpy_collection, name::String)
    for item in bpy_collection
        if item[:name] == name
            return item
        end
    end
    error("No item $name found")
end

function connect!(client::BlenderClient, host, port)
    conn = rpyc.classic[:connect](host, port)
    client.bpy = conn[:modules][:bpy]
    client.scene = client.bpy[:context][:scene]
    client.objects = Dict{String,Object}()
    for object in client.bpy[:data][:objects]
        client.objects[object[:name]] = Object(object)
    end
end

function get_camera(client::BlenderClient)
    client.objects["Camera"]
end

function get_object_names(client::BlenderClient)
    keys(client.objects)
end

function get_object(client::BlenderClient, name)
    client.objects[name]
end

function set_resolution!(client::BlenderClient, x, y)
    client.scene[:render][:resolution_x] = x
    client.scene[:render][:resolution_y] = y
end

function _add_object!(client::BlenderClient, pyobj::PyObject, name::String)
    println(client.objects)
    if haskey(client.objects, name)
        error("Object with name $name already exists")
    end
    pyobj[:name] = name
    client.objects[name] = Object(pyobj)
    client.objects[name]
end

function add_plane!(client::BlenderClient, name::String, loc::Point3, rot::Point3, scale::Point3)
    client.bpy[:ops][:mesh][:primitive_plane_add]()
    pyobj = client.bpy[:context][:object]
    pyobj[:location] = tup(loc)
    pyobj[:rotation_euler] = tup(rot)
    pyobj[:scale] = tup(scale)
    _add_object!(client, pyobj, name)
end

function delete_object!(client::BlenderClient, name::String)
    client.objects[name][:delete]()
    delete!(client.objects, name)
end

function render(client::BlenderClient, filepath)
    client.scene[:render][:filepath] = filepath
    client.bpy[:ops][:render][:render](write_still=true)
end

client = BlenderClient()
connect!(client, "localhost", 59892)
set_internal_render_engine!(client)
set_resolution!(client, 500, 500)
camera = get_object(client, "Camera")
set_location!(camera, Point3(0, -8.5, 5))
set_rotation_euler!(camera, Point3(pi/3., 0, 0))
add_plane!(client, "wall", Point3(0,10,0), Point3(pi/3.,0,0), Point3(20,20,20))
render(client, "$(pwd())/test_depth.png")
