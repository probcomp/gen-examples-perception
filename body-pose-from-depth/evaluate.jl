using DataFrames
using Gen
using GenTF
import TensorFlow
tf = TensorFlow
using CSV

include("model.jl")
include("inference.jl")
include("neural_proposal.jl")

function add_columns!(pose::BodyPose, df::DataFrame)
    df[:rot_z] = pose.rotation.z
    df[:elbow_r_loc_x] = pose.elbow_r_loc.x
    df[:elbow_r_loc_y] = pose.elbow_r_loc.y
    df[:elbow_r_loc_z] = pose.elbow_r_loc.z
    df[:elbow_l_loc_x] = pose.elbow_l_loc.x
    df[:elbow_l_loc_y] = pose.elbow_l_loc.y
    df[:elbow_l_loc_z] = pose.elbow_l_loc.z
    df[:elbow_l_rot_z] = pose.elbow_l_rot.z
    df[:elbow_r_rot_z] = pose.elbow_r_rot.z
    df[:hip_loc_x] = pose.hip_loc.x
    df[:hip_loc_y] = pose.hip_loc.y
    df[:hip_loc_z] = pose.hip_loc.z
    df[:heel_r_loc_x] = pose.heel_r_loc.x
    df[:heel_r_loc_y] = pose.heel_r_loc.y
    df[:heel_r_loc_z] = pose.heel_r_loc.z
    df[:heel_l_loc_x] = pose.heel_l_loc.x
    df[:heel_l_loc_y] = pose.heel_l_loc.y
    df[:heel_l_loc_z] = pose.heel_l_loc.z
end

function square_error(pose1::BodyPose, pose2::BodyPose)
    err = 0.
    err += norm(pose1.rotation - pose2.rotation)^2
    err += norm(pose1.elbow_r_loc - pose2.elbow_r_loc)^2
    err += norm(pose1.elbow_l_loc - pose2.elbow_l_loc)^2
    err += norm(pose1.elbow_r_rot - pose2.elbow_r_rot)^2
    err += norm(pose1.elbow_l_rot - pose2.elbow_l_rot)^2
    err += norm(pose1.hip_loc - pose2.hip_loc)^2
    err += norm(pose1.heel_r_loc - pose2.heel_r_loc)^2
    err += norm(pose1.heel_l_loc - pose2.heel_l_loc)^2
    return err
end

function evaluate_single(ground_truth, percept,
                  inference_programs::Dict{String,InferenceProgram},
                  replicates::Int)
    keys = String[]
    square_errors = Float64[]
    elapsed = Float64[]
    for (key, program) in inference_programs
        for rep=1:replicates
            println("key: $key, replicate : $rep of $replicates")
            start = time()
            latents = infer(program, percept)
            push!(elapsed, time() - start)
            push!(square_errors, square_error(latents, ground_truth))
            push!(keys, key)
        end
    end
    df = DataFrame()
    df[:elapsed] = elapsed
    df[:square_error] = square_errors
    df[:key] = keys

    # adds columns, with the same value for all rows
    add_columns!(ground_truth, df) 
    return df
end

function evaluate_multiple(scene_model, renderer,
                  inference_programs::Dict{String,InferenceProgram},
                  num_percepts::Int, replicates::Int)
    dfs = Vector{DataFrame}(undef, num_percepts)
    for i=1:num_percepts
        println("percept: $i of $num_percepts")
        # NOTE: could be modified to use real-world labelled training data
        ground_truth = sample(scene_model)
        percept = render(renderer, ground_truth)
        dfs[i] = evaluate_single(ground_truth, percept, inference_programs,
                                 replicates)
        println(dfs[i])
    end
    df = vcat(dfs...)
end


# load large NN
arch = NetworkArchitecture(32, 32, 64, 1024)
proposal_large = make_neural_proposal(arch)
session = init_session!(proposal_large.network)
params_fname = "params_arch_32_32_64_128-59902-36.jld"
as_default(GenTF.get_graph(proposal_large.network)) do
    saver = tf.train.Saver()
    tf.train.restore(saver, session, params_fname)
end

Gen.load_generated_functions()

blender = "blender"
model = "HumanKTH.decimated.blend"
renderer = BodyPoseDepthRenderer(width, height, blender, model, 59897)

import Random
Random.seed!(1)

# generate table of accuracy and runtimes
inference_programs = Dict{String,InferenceProgram}()
for num_importance_samples in [1, 10, 100]#, 100, 1000, 10000]

    inference_programs["sir-prior-$num_importance_samples"] = SIRPrior(
        renderer, num_importance_samples)

    #inference_programs["sir-nn-small-$num_importance_samples"] = SIRNN(
        #renderer, num_importance_samples, proposal_small.neural_proposal)

    inference_programs["sir-nn-large-$num_importance_samples"] = SIRNN(
        renderer, num_importance_samples, proposal_large.neural_proposal)
end
df = evaluate_multiple(BodyPoseSceneModel(), renderer, inference_programs, 1, 100)
CSV.write("evaluation.csv", df)


