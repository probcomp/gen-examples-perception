using Gen: categorical
import GenTF

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

# Raw neural network mean predictions

struct NNMeanPredictor <: InferenceProgram
    network::GenTF.TensorFlowFunction
end

mean_of_beta(a, b) = a / (a + b)
function mode_of_beta(name, a, b)
    if a > 1 && b > 1
        println("$name; a: $a, b: $b (mode)")
        (a - 1) / (a + b - 2)
    else
        println("$name; a: $a, b: $b (mean)")
        mean_of_beta(a, b)
    end
end

function infer(program::NNMeanPredictor, image::Matrix{Float64})
    
    image_flat::Matrix{Float64} = reshape(image, 1, width * height)
    output_mat::Matrix{Float64} = GenTF.exec_tf_function(program.network, (image_flat,))
    @assert size(output_mat) == (1, num_output)
    outputs = output_mat[1,:]

    # global rotation
    rot_z = mode_of_beta("rot_z", exp(outputs[1]), exp(outputs[2]))
    rotation::Point3 = scale_rot(rot_z)

    # right elbow location
    elbow_r_loc_x = mode_of_beta("elbow_r_loc_x", exp(outputs[3]), exp(outputs[4]))
    elbow_r_loc_y = mode_of_beta("elbow_r_loc_y", exp(outputs[5]), exp(outputs[6]))
    elbow_r_loc_z = mode_of_beta("elbow_r_loc_z", exp(outputs[7]), exp(outputs[8]))
    elbow_r_loc::Point3 = scale_elbow_r_loc(elbow_r_loc_x, elbow_r_loc_y, elbow_r_loc_z)

    # left elbow location
    elbow_l_loc_x = mode_of_beta("elbow_l_loc_x", exp(outputs[11]), exp(outputs[12]))
    elbow_l_loc_y = mode_of_beta("elbow_l_loc_y", exp(outputs[13]), exp(outputs[14]))
    elbow_l_loc_z = mode_of_beta("elbow_l_loc_z", exp(outputs[15]), exp(outputs[16]))
    elbow_l_loc::Point3 = scale_elbow_l_loc(elbow_l_loc_x, elbow_l_loc_y, elbow_l_loc_z)

    # right elbow rotation
    elbow_r_rot_z = mode_of_beta("elbow_r_rot_z", exp(outputs[9]), exp(outputs[10]))
    elbow_r_rot::Point3 = scale_elbow_r_rot(elbow_r_rot_z)

    # left elbow rotation
    elbow_l_rot_z = mode_of_beta("elbow_l_rot_z", exp(outputs[17]), exp(outputs[18]))
    elbow_l_rot::Point3 = scale_elbow_l_rot(elbow_l_rot_z)

    # hip
    hip_loc_z = mode_of_beta("hip_loc_z", exp(outputs[19]), exp(outputs[20]))
    hip_loc::Point3 = scale_hip_loc(hip_loc_z)

    # right heel
    heel_r_loc_x = mode_of_beta("heel_r_loc_x", exp(outputs[21]), exp(outputs[22]))
    heel_r_loc_y = mode_of_beta("heel_r_loc_y", exp(outputs[23]), exp(outputs[24]))
    heel_r_loc_z = mode_of_beta("heel_r_loc_z", exp(outputs[25]), exp(outputs[26]))
    heel_r_loc::Point3 = scale_heel_r_loc(heel_r_loc_x, heel_r_loc_y, heel_r_loc_z)

    # left heel
    heel_l_loc_x = mode_of_beta("heel_l_loc_x", exp(outputs[27]), exp(outputs[28]))
    heel_l_loc_y = mode_of_beta("heel_l_loc_y", exp(outputs[29]), exp(outputs[30]))
    heel_l_loc_z = mode_of_beta("heel_l_loc_z", exp(outputs[31]), exp(outputs[32]))
    heel_l_loc::Point3 = scale_heel_l_loc(heel_l_loc_x, heel_l_loc_y, heel_l_loc_z)

    return BodyPose(
        rotation,
        elbow_r_loc,
        elbow_l_loc,
        elbow_r_rot,
        elbow_l_rot,
        hip_loc,
        heel_r_loc,
        heel_l_loc)
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

