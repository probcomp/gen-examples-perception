using DataFrames
using Gen
using CSV

include("model.jl")
include("inference.jl")

function quant_evaluate_single(ground_truth, percept,
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

function quant_evaluate_multiple(scene_model, renderer,
                  inference_programs::Dict{String,InferenceProgram},
                  num_percepts::Int, replicates::Int)
    dfs = Vector{DataFrame}(undef, num_percepts)
    for i=1:num_percepts
        println("percept: $i of $num_percepts")
        # NOTE: could be modified to use real-world labelled training data
        ground_truth = sample(scene_model)
        percept = render(renderer, ground_truth)
        dfs[i] = quant_evaluate_single(ground_truth, percept, inference_programs,
                                 replicates)
    end
    df = vcat(dfs...)
end

# do experiments

Gen.load_generated_functions()
renderer = BodyPoseRenderer(128, 128, "localhost", 59893)

# generate table of accuracy and runtimes
programs = Dict{String,InferenceProgram}()
for num_importance_samples in [1, 10, 100, 1000]
    #programs["sir-prior-$num_importance_samples"] = SIRPrior(renderer, num_importance_samples)
    progams["sir-nn-$num_importance_samples"] = SIRNN(
        renderer, num_importance_samples)
end
df = quant_evaluate_multiple(BodyPoseSceneModel(), renderer, programs, 10, 10)
CSV.write("sir-nn.csv", df)
