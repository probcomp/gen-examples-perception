import JLD
using PyCall

@pyimport matplotlib.pyplot as plt

include("model.jl")

function abs_diff(a::Trace, b::Trace)
    a_pose = trace_to_pose(a)
    b_pose = trace_to_pose(b)
    diff = 0.
    diff += norm(a_pose.rotation - b_pose.rotation)
    diff += norm(a_pose.arm_elbow_r_location - b_pose.arm_elbow_r_location)
    diff += norm(a_pose.arm_elbow_l_location - b_pose.arm_elbow_l_location)
    diff += norm(a_pose.arm_elbow_r_rotation - b_pose.arm_elbow_r_rotation)
    diff += norm(a_pose.arm_elbow_l_rotation - b_pose.arm_elbow_l_rotation)
    diff += norm(a_pose.hip_location - b_pose.hip_location)
    diff += norm(a_pose.heel_r_location - b_pose.heel_r_location)
    diff += norm(a_pose.heel_l_location - b_pose.heel_l_location)
    diff
end

# render ground truth wireframe
ground_truth_trace = JLD.load("ground_truth_data.jld")["trace"]
render_wireframe(ground_truth_trace, "results/ground_truth.png")

# render results of importance sampling
importance_sampling_data = JLD.load("importance_sampling.jld")
importance_sampling_traces = importance_sampling_data["traces"]
importance_sampling_runtimes = importance_sampling_data["runtimes"]
importance_sampling_error = Dict()
for n in [1, 10, 100, 1000]

    # prior proposal
    traces = importance_sampling_traces[("prior", n)]
    diffs = Float64[]
    for (i, trace) in enumerate(traces)
        push!(diffs, abs_diff(ground_truth_trace, trace))
        render_wireframe(trace, @sprintf("results/is.prior.n%04d.%03d.png", n, i))
    end
    println("IS (prior, $n); median runtime: $(median(importance_sampling_runtimes[("prior", n)])), error: $(mean(diffs))")
    importance_sampling_error[("prior", n)] = mean(diffs)

    # deep learning proposal
    traces = importance_sampling_traces[("dl", n)]
    diffs = Float64[]
    for (i, trace) in enumerate(traces)
        push!(diffs, abs_diff(ground_truth_trace, trace))
        render_wireframe(trace, @sprintf("results/is.dl.n%04d.%03d.png", n, i))
    end
    println("IS (dl, $n); median runtime: $(median(importance_sampling_runtimes[("dl", n)])), error: $(mean(diffs))")
    importance_sampling_error[("dl", n)] = mean(diffs)
end

# render results of MCMC
mcmc_data = JLD.load("mcmc.jld")
mcmc_traces = mcmc_data["traces"]
mcmc_runtimes = mcmc_data["runtimes"]
mcmc_error = Dict()
for n in [1, 10, 100, 1000]

    # prior initialization
    traces = mcmc_traces[("prior-init", n)]
    diffs = Float64[]
    for (i, trace) in enumerate(traces)
        push!(diffs, abs_diff(ground_truth_trace, trace))
        render_wireframe(trace, @sprintf("results/mcmc.prior-init.n%04d.%03d.png", n, i))
    end
    println("MCMC (prior-init, $n); median runtime: $(median(mcmc_runtimes[("prior-init", n)])), error: $(mean(diffs))")
    mcmc_error[("prior-init", n)] = mean(diffs)

    # deep learning initialization
    traces = mcmc_traces[("dl-init", n)]
    diffs = Float64[]
    for (i, trace) in enumerate(traces)
        push!(diffs, abs_diff(ground_truth_trace, trace))
        render_wireframe(trace, @sprintf("results/mcmc.dl-init.n%04d.%03d.png", n, i))
    end
    println("MCMC (dl-init, $n); median runtime: $(median(mcmc_runtimes[("dl-init", n)])), error: $(mean(diffs))")
    mcmc_error[("dl-init", n)] = mean(diffs)
end

# plot the error
plt.figure()
plt.scatter(
    [median(importance_sampling_runtimes[("prior", n)]) for n in [1, 10, 100, 1000]],
    [importance_sampling_error[("prior", n)] for n in [1, 10, 100, 1000]],
    label="IS prior")
plt.scatter(
    [median(importance_sampling_runtimes[("dl", n)]) for n in [1, 10, 100, 1000]],
    [importance_sampling_error[("dl", n)] for n in [1, 10, 100, 1000]],
    label="IS dl")
plt.scatter(
    [median(mcmc_runtimes[("prior-init", n)]) for n in [1, 10, 100, 1000]],
    [mcmc_error[("prior-init", n)] for n in [1, 10, 100, 1000]],
    label="MCMC prior-init")
plt.scatter(
    [median(mcmc_runtimes[("dl-init", n)]) for n in [1, 10, 100, 1000]],
    [mcmc_error[("dl-init", n)] for n in [1, 10, 100, 1000]],
    label="MCMC dl-init")
plt.gca()[:set_xscale]("log")
plt.legend()
plt.savefig("results.pdf")
