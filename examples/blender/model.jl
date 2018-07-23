using GenLite
Gen = GenLite

using FileIO


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

# Provides Point3
include("blender_depth_client.jl")

function make_blender_client()
    client = BlenderClient()
    connect!(client, "localhost", 59892)
    setup_for_depth!(client)
    set_resolution!(client, 200, 200)
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

const default_body_pose = get_body_pose(client)

function render(pose::BodyPose)
    tmp = tempname() * ".png"
    set_body_pose!(client, pose)
    render(client, tmp)
    # TODO delete the tmp file
    convert(Matrix{Float64}, load(tmp))
end

# ensure the convert method gets compiled
# TODO debug this, may be a bug in FileIO
tmp = tempname() * ".png"
set_body_pose!(client, default_body_pose)
render(client, tmp)
img = load(tmp)
mat = convert(Matrix{Float64}, img)
println(typeof(mat))

    
model = @generative function ()

    # whole body rotation
    rotation = Point3(0., 0.,
        @rand(uniform(-pi/2, pi/2), "rotation"))

    # right elbow
    arm_elbow_r_location = Point3(
        @rand(uniform(-1, 0), "arm_elbow_r_location_dx"),
        @rand(uniform(-1, 1), "arm_elbow_r_location_dy"),
        @rand(uniform(-1, 1), "arm_elbow_r_location_dz"))

    arm_elbow_r_rotation = Point3(0., 0.,
        @rand(uniform(0, 2*pi), "arm_elbow_r_rotation_dz"))

    # left elbow
    arm_elbow_l_location = Point3(
        @rand(uniform(0, 1), "arm_elbow_l_location_dx"),
        @rand(uniform(-1, 1), "arm_elbow_l_location_dy"),
        @rand(uniform(-1, 1), "arm_elbow_l_location_dz"))

    arm_elbow_l_rotation = Point3(0., 0.,
        @rand(uniform(0, 2*pi), "arm_elbow_l_rotation_dz"))

    # hip
    hip_location = Point3(0., 0.,
        @rand(uniform(-0.35, 0), "hip_location_dz"))

    # right heel
    heel_r_location = Point3(
        @rand(uniform(-0.45, 0.1), "heel_r_location_dx"),
        @rand(uniform(-1.0, 0.5), "heel_r_location_dy"),
        @rand(uniform(-0.2, 0.2), "heel_r_location_dz"))

    # left heel
    heel_l_location = Point3(
        @rand(uniform(-0.1, 0.45), "heel_l_location_dx"),
        @rand(uniform(-1.0, 0.5), "heel_l_location_dy"),
        @rand(uniform(-0.2, 0.2), "heel_l_location_dz"))

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
    img = render(pose)

    # add speckle
    noise = 0.5
    @rand(noisy_matrix(img, noise), "image")
end

for i=1:100
    println("simulation $i")
    (trace, _, image) = simulate(model, ())
    output_filename = @sprintf("simulated_%03d.png", i)
    FileIO.save(output_filename, map(ImageCore.clamp01, image))
end
