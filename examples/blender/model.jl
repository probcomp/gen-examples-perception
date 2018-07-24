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
    connect!(client, "localhost", 59892)
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

function render(pose::BodyPose)
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

rescale(value, min, max) = min + (max - min) * value
    
model = @generative function ()

    # whole body rotation
    rotation = Point3(0., 0.,
        rescale(@rand(uniform(0, 1), "rotation"), -pi/4, pi/4))

    # right elbow
    arm_elbow_r_location = Point3(
        rescale(@rand(uniform(0, 1), "arm_elbow_r_location_dx"), -1, 0),
        rescale(@rand(uniform(0, 1), "arm_elbow_r_location_dy"), -1, 1),
        rescale(@rand(uniform(0, 1), "arm_elbow_r_location_dz"), -1, 1))

    arm_elbow_r_rotation = Point3(0., 0.,
        rescale(@rand(uniform(0, 1), "arm_elbow_r_rotation_dz"), 0, 2*pi))

    # left elbow
    arm_elbow_l_location = Point3(
        rescale(@rand(uniform(0, 1), "arm_elbow_l_location_dx"), 0, 1),
        rescale(@rand(uniform(0, 1), "arm_elbow_l_location_dy"), -1, 1),
        rescale(@rand(uniform(0, 1), "arm_elbow_l_location_dz"), -1, 1))

    arm_elbow_l_rotation = Point3(0., 0.,
        rescale(@rand(uniform(0, 1), "arm_elbow_l_rotation_dz"), 0, 2*pi))

    # hip
    hip_location = Point3(0., 0.,
        rescale(@rand(uniform(0, 1), "hip_location_dz"), -0.35, 0))

    # right heel
    heel_r_location = Point3(
        rescale(@rand(uniform(0, 1), "heel_r_location_dx"), -0.45, 0.1),
        rescale(@rand(uniform(0, 1), "heel_r_location_dy"), -1, 0.5),
        rescale(@rand(uniform(0, 1), "heel_r_location_dz"), -0.2, 0.2))

    # left heel
    heel_l_location = Point3(
        rescale(@rand(uniform(0, 1), "heel_l_location_dx"), -0.1, 0.45),
        rescale(@rand(uniform(0, 1), "heel_l_location_dy"), -1, 0.5),
        rescale(@rand(uniform(0, 1), "heel_l_location_dz"), -0.2, 0.2))

    pose_difference = BodyPose(
        rotation,
        arm_elbow_r_location,
        arm_elbow_l_location,
        arm_elbow_r_rotation,
        arm_elbow_l_rotation,
        hip_location,
        heel_r_location,
        heel_l_location)

    pose = default_body_pose + pose_difference

    # render
    ground_truth_image = render(pose)

    # blur it
    blur_amount = 1
    blurred_image = ImageFiltering.imfilter(ground_truth_image,
                    ImageFiltering.Kernel.gaussian(blur_amount))
    # add speckle
    noise = 0.1
    observable_image = @rand(noisy_matrix(blurred_image, noise), "image")

    (ground_truth_image, blurred_image, observable_image)
end

