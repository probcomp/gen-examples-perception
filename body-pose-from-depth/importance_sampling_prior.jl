using DataFrames
using Gen
using Gen: categorical
using CSV

include("model.jl")
include("evaluation.jl")

struct SIRPrior <: InferenceProgram
    renderer::BodyPoseRenderer
    num_importance_samples::Int
end

function infer(program::SIRPrior, image::Matrix{Float64})
    observations = DynamicChoiceTrie()
    observations[:image] = image
    (traces, log_normalized_weights) = importance_sampling_prior(
        generative_model, (program.renderer,), StaticChoiceTrie(observations),
        program.num_importance_samples)
    dist = exp.(log_normalized_weights)
    idx = categorical(dist)
    choices = get_internal_node(get_choices(traces[idx]), :pose)
    return BodyPose(choices)
end

Gen.load_generated_functions()

renderer = BodyPoseRenderer(64, 64, "localhost", 59893)

# show samples using wireframe rendering
# show the ground truth, alongside results for each algorithm.


# generate table of accuracy and runtimes

programs = Dict{String,InferenceProgram}()
for num_importance_samples in [10000]#[1, 10, 100, 1000]
    programs["sir-prior-$num_importance_samples"] = SIRPrior(renderer, num_importance_samples)
end
df = evaluate(BodyPoseSceneModel(), renderer, programs, 10, 10)
println(df)
CSV.write("sir-prior-2.csv", df)
