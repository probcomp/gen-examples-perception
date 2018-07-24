using GenLite
Gen = GenLite

import FileIO
import ImageFiltering


###########################
# primitive distributions #
###########################

struct NoisyMatrix <: Distribution{Matrix{Float64}} end
const noisy_matrix = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Matrix{Float64},
                    args::Tuple{Matrix{U},T}) where {U<:Real,T<:Real}
    (mu, noise) = args
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function Base.rand(::NoisyMatrix, args::Tuple{Matrix{U},T}) where {U<:Real,T<:Real}
    (mu, noise) = args
    mat = copy(mu)
    (w, h) = size(mu)
    for i=1:w
        for j=1:h
            mat[i, j] = mu[i, j] + randn() * noise
        end
    end
    mat
end


#########
# model #
#########

# Provides BlenderClient and Point3
include("blender_depth_client.jl")

const width = 128
const height = 128
const num_pixels = width * height

function make_blender_client()
    client = BlenderClient()
    connect!(client, "localhost", 59893)
    setup_for_depth!(client)
    set_resolution!(client, width, height)
    set_object_location!(client, "Camera", Point3(0, -8.5, 5))
    set_object_rotation_euler!(client, "Camera", Point3(pi/3., 0, 0))
    add_plane!(client, "background", Point3(0,4,0), Point3(pi/3.,0,0), Point3(20,20,20))
    set_object_location!(client, RIG, Point3(0, 0, 0))
    set_object_rotation_euler!(client, RIG, Point3(0, 0, 0))
    set_object_scale!(client, RIG, Point3(3, 3, 3))
    add_plane!(client, "nearplane", Point3(-2,-4,0), Point3(pi/3.,0,0), Point3(0.1,0.1,0.1))
    client
end

const client = make_blender_client()

const default_body_pose = BodyPose(
    Point3(0.0, 0.0, 0.0),
    Point3(0.0, 0.0, 0.0),
    Point3(0.0, 0.0, 0.0),
    Point3(0.0, 0.0, 0.0),
    Point3(0.0, 0.0, 0.0),
    Point3(0.0, 0.0, 0.0),
    Point3(0.0, 0.0, 0.0),
    Point3(0.0, 0.0, 0.0))

setup_for_depth!(client)
set_resolution!(client, width, height)

function render_depth(pose::BodyPose)
    tmp = tempname() * ".png"
    set_body_pose!(client, pose)
    render(client, tmp)
    img = FileIO.load(tmp)
    rm(tmp)
    convert(Matrix{Float64}, img)
end

# ensure the convert method gets compiled
# TODO debug this, may be a bug in FileIO
tmp = tempname() * ".png"
set_body_pose!(client, default_body_pose)
render(client, tmp)
img = FileIO.load(tmp)
rm(tmp)
mat = convert(Matrix{Float64}, img)
println(typeof(mat))

# rescale values from [0, 1] to another interval
scale(value, min, max) = min + (max - min) * value
rotation(z) = Point3(0., 0., scale(z, -pi/4, pi/4))
arm_elbow_r_location(x, y, z) = Point3(scale(x, -1, 0), scale(y, -1, 1), scale(z, -1, 1))
arm_elbow_r_rotation(z) = Point3(0., 0., scale(z, 0, 2*pi))
arm_elbow_l_location(x, y, z) = Point3(scale(x, 0, 1), scale(y, -1, 1), scale(z, -1, 1))
arm_elbow_l_rotation(z) = Point3(0., 0., scale(z, 0, 2*pi))
hip_location(z) = Point3(0., 0., scale(z, -0.35, 0))
heel_r_location(x, y, z) = Point3(scale(x, -0.45, 0.1), scale(y, -1, 0.5), scale(z, -0.2, 0.2))
heel_l_location(x, y, z) = Point3(scale(x, -0.1, 0.45), scale(y, -1, 0.5), scale(z, -0.2, 0.2))

model = @generative function ()

    pose = BodyPose(
        rotation(
            @rand(uniform(0, 1), "rotation")),
        arm_elbow_r_location(
            @rand(uniform(0, 1), "arm_elbow_r_location_dx"),
            @rand(uniform(0, 1), "arm_elbow_r_location_dy"),
            @rand(uniform(0, 1), "arm_elbow_r_location_dz")),
        arm_elbow_l_location(
            @rand(uniform(0, 1), "arm_elbow_l_location_dx"),
            @rand(uniform(0, 1), "arm_elbow_l_location_dy"),
            @rand(uniform(0, 1), "arm_elbow_l_location_dz")),
        arm_elbow_r_rotation(
            @rand(uniform(0, 1), "arm_elbow_r_rotation_dz")),
        arm_elbow_l_rotation(
            @rand(uniform(0, 1), "arm_elbow_l_rotation_dz")),
        hip_location(
            @rand(uniform(0, 1), "hip_location_dz")),
        heel_r_location(
            @rand(uniform(0, 1), "heel_r_location_dx"),
            @rand(uniform(0, 1), "heel_r_location_dy"),
            @rand(uniform(0, 1), "heel_r_location_dz")),
        heel_l_location(
            @rand(uniform(0, 1), "heel_l_location_dx"),
            @rand(uniform(0, 1), "heel_l_location_dy"),
            @rand(uniform(0, 1), "heel_l_location_dz")))

    # render
    depth = render_depth(pose)

    # blur it
    blur_amount = 1
    blurred = ImageFiltering.imfilter(depth,
                    ImageFiltering.Kernel.gaussian(blur_amount))
    # add speckle
    noise = 0.1
    observable = @rand(noisy_matrix(blurred, noise), "image")

    (depth, blurred, observable)
end

function trace_to_pose(t::Trace)
    BodyPose(
        rotation(
            t["rotation"]),
        arm_elbow_r_location(
            t["arm_elbow_r_location_dx"],
            t["arm_elbow_r_location_dy"],
            t["arm_elbow_r_location_dz"]),
        arm_elbow_l_location(
            t["arm_elbow_l_location_dx"],
            t["arm_elbow_l_location_dy"],
            t["arm_elbow_l_location_dz"]),
        arm_elbow_r_rotation(
            t["arm_elbow_r_rotation_dz"]),
        arm_elbow_l_rotation(
            t["arm_elbow_l_rotation_dz"]),
        hip_location(
            t["hip_location_dz"]),
        heel_r_location(
            t["heel_r_location_dx"],
            t["heel_r_location_dy"],
            t["heel_r_location_dz"]),
        heel_l_location(
            t["heel_l_location_dx"],
            t["heel_l_location_dy"],
            t["heel_l_location_dz"]))
end

function render_wireframe(trace::Trace, filepath)
    pose = trace_to_pose(trace)
    set_body_pose!(client, pose)
    setup_for_wireframe!(client)
    set_resolution!(client, 400, 400)
    render(client, filepath)
end
