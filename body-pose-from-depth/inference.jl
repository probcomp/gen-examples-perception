using Gen: categorical

abstract type InferenceProgram end

# SIR from the prior

struct SIRPrior <: InferenceProgram
    renderer::BodyPoseDepthRenderer
    num_importance_samples::Int
end

function infer(program::SIRPrior, image::Matrix{Float64})
    observations = DynamicChoiceTrie()
    observations[:image] = image
    (trace, _) = importance_resampling(generative_model, (program.renderer,),
         StaticChoiceTrie(observations), program.num_importance_samples)
    choices = get_internal_node(get_choices(trace), :pose)
    return BodyPose(choices)
end


# SIR using neural network proposal

struct SIRNN <: InferenceProgram
    renderer::BodyPoseDepthRenderer
    num_importance_samples::Int
    proposal::Generator
end

function infer(program::SIRNN, image::Matrix{Float64})
    observations = DynamicChoiceTrie()
    observations[:image] = image
    (trace, _) = importance_resampling(generative_model, (program.renderer,),
         StaticChoiceTrie(observations),
         program.proposal, (image,),
         program.num_importance_samples)
    choices = get_internal_node(get_choices(trace), :pose)
    return BodyPose(choices)
end

# MCMC

struct MCMC <: InferenceProgram
    renderer::BodyPoseDepthRenderer
    num_steps::Int
end

@compiled @gen function rot_proposal_inner(prev)
    @addr(uniform(0, 1), :rot_z)
end

@compiled @gen function rot_proposal(prev)
    @addr(rot_proposal_inner(prev), :pose)
end

rot_move(trace) = mh(generative_model, rot_proposal, (), trace)

@compiled @gen function elbow_r_proposal_inner(prev)
    @addr(uniform(0, 1), :elbow_r_loc_x)
    @addr(uniform(0, 1), :elbow_r_loc_y)
    @addr(uniform(0, 1), :elbow_r_loc_z)
    @addr(uniform(0, 1), :elbow_r_rot_z)
end

@compiled @gen function elbow_r_proposal(prev)
    @addr(elbow_r_proposal_inner(prev), :pose)
end

elbow_r_move(trace) = mh(generative_model, elbow_r_proposal, (), trace)

@compiled @gen function elbow_l_proposal_inner(prev)
    @addr(uniform(0, 1), :elbow_l_loc_x)
    @addr(uniform(0, 1), :elbow_l_loc_y)
    @addr(uniform(0, 1), :elbow_l_loc_z)
    @addr(uniform(0, 1), :elbow_l_rot_z)
end

@compiled @gen function elbow_l_proposal(prev)
    @addr(elbow_l_proposal_inner(prev), :pose)
end

elbow_l_move(trace) = mh(generative_model, elbow_l_proposal, (), trace)

@compiled @gen function hip_proposal_inner(prev)
    @addr(uniform(0, 1), :hip_loc_z)
end

@compiled @gen function hip_proposal(prev)
    @addr(hip_proposal_inner(prev), :pose)
end

hip_move(trace) = mh(generative_model, hip_proposal, (), trace)

@compiled @gen function heel_r_proposal_inner(prev)
    @addr(uniform(0, 1), :heel_r_loc_x)
    @addr(uniform(0, 1), :heel_r_loc_y)
    @addr(uniform(0, 1), :heel_r_loc_z)
end

@compiled @gen function heel_r_proposal(prev)
    @addr(heel_r_proposal_inner(prev), :pose)
end


heel_r_move(trace) = mh(generative_model, heel_r_proposal, (), trace)

@compiled @gen function heel_l_proposal_inner(prev)
    @addr(uniform(0, 1), :heel_l_loc_x)
    @addr(uniform(0, 1), :heel_r_loc_y)
    @addr(uniform(0, 1), :heel_r_loc_z)
end

@compiled @gen function heel_l_proposal(prev)
    @addr(heel_l_proposal_inner(prev), :pose)
end

heel_l_move(trace) = mh(generative_model, heel_l_proposal, (), trace)

function infer(program::MCMC, image::Matrix{Float64})
    observations = DynamicChoiceTrie()
    observations[:image] = image
    static_observations = StaticChoiceTrie(observations)
    (trace, _) = generate(generative_model, (program.renderer,), static_observations)
    for iter=1:program.num_steps
        trace = rot_move(trace)
        trace = elbow_r_move(trace)
        trace = elbow_l_move(trace)
        trace = hip_move(trace)
        trace = heel_r_move(trace)
        trace = heel_l_move(trace)
    end
    choices = get_internal_node(get_choices(trace), :pose)
    return BodyPose(choices)
end

