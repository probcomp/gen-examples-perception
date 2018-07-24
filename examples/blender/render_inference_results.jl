import JLD

include("model.jl")

# render ground truth wireframe
ground_truth_trace = JLD.load("ground_truth_data.jld")["trace"]
render_wireframe(ground_truth_trace, "results/ground_truth.png")

# render results of importance sampling
importance_sampling_data = JLD.load("importance_sampling.jld")
importance_sampling_traces = importance_sampling_data["traces"]
importance_sampling_runtimes = importance_sampling_data["runtimes"]
for n in [1, 10]
    traces = importance_sampling_traces[("prior", n)]
    for (i, trace) in enumerate(traces)
        render_wireframe(trace, @sprintf("results/is.prior.n%04d.%03d.png", n, i))
    end
    traces = importance_sampling_traces[("dl", n)]
    for (i, trace) in enumerate(traces)
        render_wireframe(trace, @sprintf("results/is.dl.n%04d.%03d.png", n, i))
    end
end

# TODO render MCMC
