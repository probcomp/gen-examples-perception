using DataFrames
using Gen
using GenTF
import TensorFlow
tf = TensorFlow
using CSV

include("model.jl")
include("inference.jl")
include("neural_proposal.jl")

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
    end
    df = vcat(dfs...)
end


# load the neural nets

#println("small arch...")
#arch_small = NetworkArchitecture(8, 8, 16, 128)
#proposal_small = make_neural_proposal(arch_small)
#session = init_session!(proposal_small.network)
#tf.as_default(GenTF.get_graph(proposal_small.network)) do
    #saver = tf.train.Saver()
    #tf.train.restore(saver, session, "params_small_arch.jld")
#end

#println("large arch...")
#arch_large = NetworkArchitecture(32, 32, 64, 1024)
#proposal_large = make_neural_proposal(arch_large)
#session = init_session!(proposal_large.network)
#tf.as_default(GenTF.get_graph(proposal_large.network)) do
    #saver = tf.train.Saver()
    #tf.train.restore(saver, session, "params_large_arch.jld")
#end

Gen.load_generated_functions()
renderer = BodyPoseRenderer(128, 128, "localhost", 59893)

# generate table of accuracy and runtimes
inference_programs = Dict{String,InferenceProgram}()
for num_importance_samples in [1, 10, 100, 1000, 10000]

    inference_programs["sir-prior-$num_importance_samples"] = SIRPrior(
        renderer, num_importance_samples)

    #inference_programs["sir-nn-small-$num_importance_samples"] = SIRNN(
        #renderer, num_importance_samples, proposal_small.neural_proposal)

    #inference_programs["sir-nn-large-$num_importance_samples"] = SIRNN(
        #renderer, num_importance_samples, proposal_large.neural_proposal)
end
df = evaluate_multiple(BodyPoseSceneModel(), renderer, inference_programs, 1, 1)
CSV.write("sir-prior.csv", df)


