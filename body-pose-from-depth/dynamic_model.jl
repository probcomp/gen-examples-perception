using FileIO
using Images: ImageCore

include("model.jl")

using FunctionalCollections: PersistentVector

# filtering demo

# [ ] write a program that generates a handwaving depth image sequence (e.g. of length 50 frames)
# [ ] run the raw neural network on each frame to generate a movie (should jerk around?)

# [ ] create a static movie of a single image
# [ ] generate independent samples from the deep neural network (make a movie, should jerk around a bit)


# [ ] run importance sampling inference using the deep neural network on each frame to generate a movie (should jerk around more)
# [ ] write a dynamic generative model and create a few movies sampled from it (50 frames each?)
# [ ] run particle filtering in the dynamic generative model on the static sequence, and show that we have made them less jerky
# [ ] compare particle filter with independent sampling on the handwaving example
# [ ] show that you can do initialization using the deep network, and then (many particle) filtering without it?

include("spike_and_slab.jl")

const spike_width = 0.01
const spike_mass = 0.95

@compiled @gen function step(t::Int, pose_and_image::Tuple{BodyPose,Matrix{Float64}}, params::Nothing)
    pose::BodyPose = pose_and_image[1]

    # global rotation
    rotation_x::Float64 = @addr(spike_slab(unscale_rot(pose.rotation), spike_width, spike_mass), :rot_z)
    rotation::Point3 = scale_rot(rotation_x)

    # right elbow location
    center_elbow_r_loc::Tuple{Float64,Float64,Float64} = unscale_elbow_r_loc(pose.elbow_r_loc)
    elbow_r_loc_x::Float64 = @addr(spike_slab(center_elbow_r_loc[1], spike_width, spike_mass), :elbow_r_loc_x)
    elbow_r_loc_y::Float64 = @addr(spike_slab(center_elbow_r_loc[2], spike_width, spike_mass), :elbow_r_loc_y)
    elbow_r_loc_z::Float64 = @addr(spike_slab(center_elbow_r_loc[3], spike_width, spike_mass), :elbow_r_loc_z)
    elbow_r_loc::Point3 = scale_elbow_r_loc(elbow_r_loc_x, elbow_r_loc_y, elbow_r_loc_z)
    
    # left elbow location
    center_elbow_l_loc::Tuple{Float64,Float64,Float64} = unscale_elbow_l_loc(pose.elbow_l_loc)
    elbow_l_loc_x::Float64 = @addr(spike_slab(center_elbow_l_loc[1], spike_width, spike_mass), :elbow_l_loc_x)
    elbow_l_loc_y::Float64 = @addr(spike_slab(center_elbow_l_loc[2], spike_width, spike_mass), :elbow_l_loc_y)
    elbow_l_loc_z::Float64 = @addr(spike_slab(center_elbow_l_loc[3], spike_width, spike_mass), :elbow_l_loc_z)
    elbow_l_loc::Point3 = scale_elbow_l_loc(elbow_l_loc_x, elbow_l_loc_y, elbow_l_loc_z)

    # right elbow rotation
    center_elbow_r_rot_z::Float64 = unscale_elbow_r_rot(pose.elbow_r_rot)
    elbow_r_rot_z::Float64 = @addr(spike_slab(center_elbow_r_rot_z, spike_width, spike_mass), :elbow_r_rot_z)
    elbow_r_rot::Point3 = scale_elbow_r_rot(elbow_r_rot_z)

    # left elbow rotation
    center_elbow_l_rot_z::Float64 = unscale_elbow_l_rot(pose.elbow_l_rot)
    elbow_l_rot_z::Float64 = @addr(spike_slab(center_elbow_l_rot_z, spike_width, spike_mass), :elbow_l_rot_z)
    elbow_l_rot::Point3 = scale_elbow_l_rot(elbow_l_rot_z)

    # hip
    center_hip::Float64 = unscale_hip_loc(pose.hip_loc)
    hip_loc_z::Float64 = @addr(spike_slab(center_hip, spike_width, spike_mass), :hip_loc_z)
    hip_loc::Point3 = scale_hip_loc(hip_loc_z)

    # right heel
    center_heel_r_loc::Tuple{Float64,Float64,Float64} = unscale_heel_r_loc(pose.heel_r_loc)
    heel_r_loc_x::Float64 = @addr(spike_slab(center_heel_r_loc[1], spike_width, spike_mass), :heel_r_loc_x)
    heel_r_loc_y::Float64 = @addr(spike_slab(center_heel_r_loc[2], spike_width, spike_mass), :heel_r_loc_y)
    heel_r_loc_z::Float64 = @addr(spike_slab(center_heel_r_loc[3], spike_width, spike_mass), :heel_r_loc_z)
    heel_r_loc::Point3 = scale_heel_r_loc(heel_r_loc_x, heel_r_loc_y, heel_r_loc_z)

    # left heel
    center_heel_l_loc::Tuple{Float64,Float64,Float64} = unscale_heel_l_loc(pose.heel_l_loc)
    heel_l_loc_x::Float64 = @addr(spike_slab(center_heel_l_loc[1], spike_width, spike_mass), :heel_l_loc_x)
    heel_l_loc_y::Float64 = @addr(spike_slab(center_heel_l_loc[2], spike_width, spike_mass), :heel_l_loc_y)
    heel_l_loc_z::Float64 = @addr(spike_slab(center_heel_l_loc[3], spike_width, spike_mass), :heel_l_loc_z)
    heel_l_loc::Point3 = scale_heel_l_loc(heel_l_loc_x, heel_l_loc_y, heel_l_loc_z)
    
    new_pose = BodyPose(
        rotation,
        elbow_r_loc,
        elbow_l_loc,
        elbow_r_rot,
        elbow_l_rot,
        hip_loc,
        heel_r_loc,
        heel_l_loc)

    image::Matrix{Float64} = render(renderer, new_pose)
    blurred::Matrix{Float64} = imfilter(image, Kernel.gaussian(1))
    observable::Matrix{Float64} = @addr(noisy_matrix(blurred, 0.1), :image)

    return (new_pose, observable)::Tuple{BodyPose,Matrix{Float64}}
end

hmm = markov(step)

@compiled @gen function dynamic_generative_model(renderer::BodyPoseDepthRenderer, num_steps::Int)
    init_pose::BodyPose = @addr(body_pose_model(), :pose)
    step_data::PersistentVector{Tuple{BodyPose,Matrix{Float64}}} = @addr(hmm(num_steps, (init_pose, fill(NaN, 1, 1)), nothing), :steps)
    return step_data
end

# generate some data from the dynamic model

Gen.load_generated_functions()

blender = "blender"
model = "HumanKTH.decimated.blend"
const renderer = BodyPoseDepthRenderer(width, height, blender, model, 59900)

trace = simulate(dynamic_generative_model, (renderer, 50))
step_data = get_call_record(trace).retval
images = map((tuple) -> tuple[2], step_data)

for (i, image) in enumerate(images)
    fname = @sprintf("img-%03d.png", i)
    println(fname)
    FileIO.save(fname, map(ImageCore.clamp01, image))
end


