using GenLiteTF
using TensorFlow
tf = TensorFlow

include("model.jl")
include("proposal.jl")

function make_single_site_grw_proposal(addr, width)
    @generative function()
        cur_value = @read(addr)
        @rand(normal(cur_value, width), addr)
    end
end

elbow_proposal = @generative function(side::String, width)
    cur = @read("arm_elbow_$(side)_location_dx")
    @rand(uniform(cur - width, cur + width), "arm_elbow_$(side)_location_dx")
    cur = @read("arm_elbow_$(side)_location_dy")
    @rand(uniform(cur - width, cur + width), "arm_elbow_$(side)_location_dy")
    cur = @read("arm_elbow_$(side)_location_dz")
    @rand(uniform(cur - width, cur + width), "arm_elbow_$(side)_location_dz")
    cur = @read("arm_elbow_$(side)_rotation_dz")
    @rand(uniform(cur - width, cur + width), "arm_elbow_$(side)_rotation_dz")
end

heel_proposal = @generative function(side::String, width)
    cur = @read("heel_$(side)_location_dx")
    @rand(uniform(cur - width, cur + width), "heel_$(side)_location_dx")
    cur = @read("heel_$(side)_location_dy")
    @rand(uniform(cur - width, cur + width), "heel_$(side)_location_dy")
    cur = @read("heel_$(side)_location_dz")
    @rand(uniform(cur - width, cur + width), "heel_$(side)_location_dz")
end

rotation_proposal = @generative function(width)
    cur = @read("rotation")
    @rand(uniform(cur - width, cur + width), "rotation")
end

hip_proposal = @generative function(width)
    cur = @read("hip_location_dz")
    @rand(uniform(cur - width, cur + width), "hip_location_dz")
end

function mh2_move(trace, score, proposal, proposal_args)
    (new_trace, new_score, alpha, val) = mh2(model, (), proposal, proposal_args, trace, score)
    (trace, score, accepted) = mh_accept(alpha, trace, score, new_trace, new_score)
    (trace, score, accepted, val)
end

selector = @selection function ()
        @select("arm_elbow_r_location_dx")
        @select("arm_elbow_r_location_dy")
        @select("arm_elbow_r_location_dz")
        @select("arm_elbow_r_rotation_dz")
        @select("arm_elbow_l_location_dx")
        @select("arm_elbow_l_location_dy")
        @select("arm_elbow_l_location_dz")
        @select("arm_elbow_l_rotation_dz")
        @select("hip_location_dz")
        @select("heel_r_location_dx")
        @select("heel_r_location_dy")
        @select("heel_r_location_dz")
        @select("heel_l_location_dx")
        @select("heel_l_location_dy")
        @select("heel_l_location_dz")
end

function independent_move(trace, score)
    (new_trace, new_score, alpha, val) = mh(model, (), selector, (), trace, score)
    (trace, score, accepted) = mh_accept(alpha, trace, score, new_trace, new_score)
    (trace, score, accepted, val)
end

function mcmc_moves(trace, score)
    const width = 0.2
    (trace, score, indep_accepted) = independent_move(trace, score)
    (trace, score, rotation_accepted) = mh2_move(trace, score, rotation_proposal, (width,))
    (trace, score, hip_accepted) = mh2_move(trace, score, hip_proposal, (width,))
    (trace, score, elbow_r_accepted) = mh2_move(trace, score, elbow_proposal, ("r", width))
    (trace, score, elbow_l_accepted) = mh2_move(trace, score, elbow_proposal, ("l", width))
    (trace, score, heel_r_accepted) = mh2_move(trace, score, heel_proposal, ("r", width))
    (trace, score, heel_l_accepted, val) = mh2_move(trace, score, heel_proposal, ("l", width))
    accepted = convert(Vector{Int}, [indep_accepted, rotation_accepted, hip_accepted, elbow_r_accepted, elbow_l_accepted, heel_r_accepted, heel_l_accepted])
    (trace, score, accepted, val)
end

function do_mcmc_prior_init(input_image, n::Int)

    # construct trace containing observed image
    observations = Trace()
    observations["image"] = input_image
    
    # generate initial complete trace
    (trace, score, _, _) = Gen.imp(model, (), observations)
    
    # do MCMC iterations
    scores = Vector{Float64}(n)
    for iter=1:n
        (trace, score, _, _) = mcmc_moves(trace, score)
        scores[iter] = score
    end
    
    (trace, scores)
end

function do_mcmc_dl_init(input_image, n::Int)

    # construct trace containing observed image
    observations = Trace()
    observations["image"] = input_image
    
    # generate initial complete trace
    (trace, score, _, _) = Gen.imp2(model, (), dl_proposal, (), observations)
    
    # do MCMC iterations
    scores = Vector{Float64}(n)
    for iter=1:n
        (trace, score, _, _) = mcmc_moves(trace, score)
        scores[iter] = score
    end
    
    (trace, scores)
end

function do_mcmc_inference(reps, num_iters_list, image)
    traces = Dict()
    runtimes = Dict()
    scores = Dict()
    for n in num_iters_list
        println("mcmc n=$n")
        runtimes[("prior-init", n)] = Vector{Float64}(reps)
        traces[("prior-init", n)] = Vector{Trace}(reps)
        scores[("prior-init", n)] = Vector{Vector{Float64}}(reps)
        runtimes[("dl-init", n)] = Vector{Float64}(reps)
        traces[("dl-init", n)] = Vector{Trace}(reps)
        scores[("dl-init", n)] = Vector{Vector{Float64}}(reps)
        for i=1:reps
            println("$i of $reps")
    
            println("prior-init")
            tic()
            (traces[("prior-init", n)][i], scores[("prior-init", n)][i]) = do_mcmc_prior_init(observed_image, n)
            runtimes[("prior-init", n)][i] = toq()
    
            println("dl-init")
            tic()
            (traces[("dl-init", n)][i], scores[("dl-init", n)][i]) = do_mcmc_dl_init(observed_image, n)
            runtimes[("dl-init", n)][i] = toq()
        end
        JLD.save("mcmc.jld", Dict("traces" => traces, "runtimes" => runtimes))
    end
end


tf.run(get_ambient_tf_session(), tf.global_variables_initializer())
saver = tf.train.Saver()
tf.train.restore(saver, get_ambient_tf_session(), "inference_network_params.jld")

observed_image = convert(Matrix{Float64}, FileIO.load("observed.png"))

######################
# regular experiment #
######################

do_mcmc_inference(100, [1, 3, 10, 30, 100, 300, 1000], observed_image)

#######################
# profiled experiment #
#######################

## force compilation
#do_mcmc_inference(2, [1, 2], observed_image)
#
## do profiled run
#@profile do_mcmc_inference(20, [1, 10], observed_image)
#
## save profiling data for later analysis
#using JLD
#li, lidict = Profile.retrieve()
#@save "mcmc.jlprof" li lidict
