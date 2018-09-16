using Gen: categorical

abstract type InferenceProgram end

# SIR from the prior

struct SIRPrior <: InferenceProgram
    renderer::BodyPoseRenderer
    num_importance_samples::Int
end

function infer(program::SIRPrior, image::Matrix{Float64})
    observations = DynamicChoiceTrie()
    observations[:image] = image
    (trace, _) = importance_resampling(generative_model, (program.renderer,),
         StaticChoiceTrie(observations), program.num_importance_samples)
    choices = get_internal_node(get_choices(traces[idx]), :pose)
    return BodyPose(choices)
end


# SIR using neural network proposal

struct SIRNN <: InferenceProgram
    renderer::BodyPoseRenderer
    num_importance_samples::Int
    proposal::Generator
    proposal_args::Tuple
end

function infer(program::SIRNN, image::Matrix{Float64})
    observations = DynamicChoiceTrie()
    observations[:image] = image
    (trace, _) = importance_resampling(generative_model, (program.renderer,),
         StaticChoiceTrie(observations),
         program.proposal, program.proposal_args,
         program.num_importance_samples)
    choices = get_internal_node(get_choices(traces[idx]), :pose)
    return BodyPose(choices)
end


