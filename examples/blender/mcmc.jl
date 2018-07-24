include("model.jl")

import JLD

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

# consruct observation trace
observed_image = convert(Matrix{Float64}, FileIO.load("simulated.observable.086.png"))
observations = Trace()
observations["image"] = observed_image

# generate initial complete trace
(trace, score, _, _) = Gen.imp(model, (), observations)

total_elapsed = 0.
elapsed = Float64[]
scores = Float64[]
num_iter = 10000
for iter=1:num_iter
    tic()
    (trace, score, accepted, val) = mcmc_moves(trace, score)
    this_elapsed = toq()
    total_elapsed += this_elapsed
    push!(elapsed, total_elapsed)
    (ground_truth_image, blurred_image, observable_image) = val 
    push!(scores, score)
    iter_str = @sprintf("iter: %05d", iter)
    score_str = @sprintf("score: %0.2f", score)
    accepted_str = "accepted: $accepted"
    println(iter_str * " " * accepted_str * " " * score_str)
    if iter % 10 == 0
        output_filename = @sprintf("mcmc.ground_truth.%05d.png", iter)
        FileIO.save(output_filename, map(ImageCore.clamp01, ground_truth_image))
    end
    if iter % 1000 == 0
        JLD.save("scores.jld", Dict("scores" => scores, "elapsed" => elapsed))
    end
end
