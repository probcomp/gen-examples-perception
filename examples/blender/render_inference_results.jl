import JLD

include("model.jl")

# render ground truth wireframe
ground_truth_trace = JLD.load("ground_truth_data.jld")["trace"]
render_wireframe(ground_truth_trace, "results/ground_truth.png")

# render results of importance sampling
importance_sampling_data = JLD.load("importance_sampling.jld")
importance_sampling_traces = importance_sampling_data["traces"]
importance_sampling_runtimes = importance_sampling_data["runtimes"]
for n in [1, 10, 100, 1000]
    traces = importance_sampling_traces[("prior", n)]
    for (i, trace) in enumerate(traces)
        render_wireframe(trace, @sprintf("results/is.prior.n%04d.%03d.png", n, i))
    end
    traces = importance_sampling_traces[("dl", n)]
    for (i, trace) in enumerate(traces)
        render_wireframe(trace, @sprintf("results/is.dl.n%04d.%03d.png", n, i))
    end
end

# render results of MCMC
mcmc_data = JLD.load("mcmc.jld")
mcmc_traces = mcmc_data["traces"]
mcmc_runtimes = mcmc_data["runtimes"]
for n in [1, 10, 100]
    traces = mcmc_traces[("prior-init", n)]
    for (i, trace) in enumerate(traces)
        render_wireframe(trace, @sprintf("results/mcmc.prior-init.n%04d.%03d.png", n, i))
    end
    traces = mcmc_traces[("dl-init", n)]
    for (i, trace) in enumerate(traces)
        render_wireframe(trace, @sprintf("results/mcmc.dl-init.n%04d.%03d.png", n, i))
    end
end
