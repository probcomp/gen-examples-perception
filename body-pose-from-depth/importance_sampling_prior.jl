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
    return BodyPose(get_choices(traces[idx]))
end

Gen.load_generated_functions()

renderer = BodyPoseRenderer(128, 128, "localhost", 59893)
programs = Dict{String,InferenceProgram}()
programs["sir-16"] = SIRPrior(renderer, 16)
df = evaluate(BodyPoseSceneModel(), renderer, programs, 10, 4)
println(df)
