using Gen: categorical
import GenTF

abstract type MovieInferenceProgram end

#########################################
# importance sampling at each iteration #
#########################################

struct MovieSIRNN <: MovieInferenceProgram
    renderer::BodyPoseDepthRenderer
    num_importance_samples::Int
    proposal::Generator
end

function infer(program::MovieSIRNN, images::Vector{Matrix{Float64}})
    poses = Vector{BodyPose}(undef, length(images))
    for (i, image) in enumerate(images)
        observations = DynamicChoiceTrie()
        observations[:image] = image
        (trace, _) = importance_resampling(generative_model, (program.renderer,),
             StaticChoiceTrie(observations),
             program.proposal, (image,),
             program.num_importance_samples)
        choices = get_internal_node(get_choices(trace), :pose)
        poses[i] = BodyPose(choices)
    end
    poses
end


######################
# particle filtering #
######################

struct NeuralParticleFiltering <: MovieInferenceProgram
    renderer::BodyPoseDepthRenderer
    num_particles::Int
    ess::Int
    init_proposal::Generator
    step_proposal::Generator
end

function infer(program::NeuralParticleFiltering, images::Vector{Matrix{Float64}})
    num_steps = length(images)

    function get_init_observations_and_proposal_args()
        observations = StaticChoiceTrie((init_image=images[1],), NamedTuple())
        proposal_args = (images[1],)
        (observations, proposal_args)
    end

    function get_step_observations_and_proposal_args(t::Int, prev_model_trace)
        @assert t > 1
        hmm_choices = DynamicChoiceTrie()
        set_internal_node!(hmm_choices, t-1, StaticChoiceTrie((image=images[t],), NamedTuple()))
        observations = StaticChoiceTrie(NamedTuple(), (steps=hmm_choices,))
        proposal_args = (images[t],t-1)
        (observations, proposal_args)
    end

    (traces, log_normalized_weights, log_ml_estimate) = particle_filter(
        dynamic_generative_model, (program.renderer,),
        num_steps, program.num_particles, program.ess, 
        get_init_observations_and_proposal_args,
        get_step_observations_and_proposal_args,
        program.init_proposal,
        program.step_proposal; verbose=true)

    idx = categorical(exp.(log_normalized_weights))
    trace = traces[idx]
    choices = get_choices(trace)
    poses = Vector{BodyPose}(undef, length(images))
    poses[1] = BodyPose(get_internal_node(choices, :init_pose))
    for i=2:length(images)
        poses[i] = BodyPose(get_internal_node(choices, :steps => (i-1)))
    end
    poses
end
