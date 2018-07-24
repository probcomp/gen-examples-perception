import JLD

using GenLiteTF
using TensorFlow
tf = TensorFlow

include("model.jl")
include("proposal.jl")

function logsumexp(arr::Vector{Float64})
    maxlog = maximum(arr)
    maxlog + log(sum(exp.(arr - maxlog)))
end

function do_importance_sampling_dl(input_image, n::Int)

    # construct trace containing observed image
    observations = Trace()
    observations["image"] = input_image

    # do importance sampling
    latent_images = Vector{Matrix{Float64}}(n)
    traces = Vector{Trace}(n)
    log_weights = Vector{Float64}(n)
    for i=1:n
        (traces[i], _, log_weights[i], _) = Gen.imp2(
            model, (), dl_proposal, (), observations)
    end

    dist = exp.(log_weights - logsumexp(log_weights))
    idx = rand(categorical, (dist,))
    traces[idx]
end

function do_importance_sampling_prior(input_image, n::Int)

    # construct trace containing observed image
    observations = Trace()
    observations["image"] = input_image

    # do importance sampling
    latent_images = Vector{Matrix{Float64}}(n)
    traces = Vector{Trace}(n)
    log_weights = Vector{Float64}(n)
    for i=1:n
        (traces[i], _, log_weights[i], _) = Gen.imp(
            model, (), observations)
    end

    dist = exp.(log_weights - logsumexp(log_weights))
    idx = rand(categorical, (dist,))
    traces[idx]
end

tf.run(get_ambient_tf_session(), tf.global_variables_initializer())
saver = tf.train.Saver()
tf.train.restore(saver, get_ambient_tf_session(), "inference_network_params.jld")

input_fname = "simulated.observable.009.png"
input_image = convert(Matrix{Float64}, FileIO.load(input_fname))
traces = Dict()
runtimes = Dict()
for n in [1, 10, 100, 1000]
    println("importance sampling n=$n")
    reps = 20
    runtimes[("prior", n)] = Vector{Float64}(reps)
    runtimes[("dl", n)] = Vector{Float64}(reps)
    traces[("prior", n)] = Vector{Trace}(reps)
    traces[("dl", n)] = Vector{Trace}(reps)
    for i=1:reps
        println("$i of $reps")

        # do importance sampling using the prior
        println("prior")
        tic()
        traces[("prior", n)][i] = do_importance_sampling_prior(input_image, n)
        runtimes[("prior", n)][i] = toq()

        # do importance sampling using the deep learning proposal
        println("deep learning")
        tic()
        traces[("dl", n)][i] = do_importance_sampling_dl(input_image, n)
        runtimes[("dl", n)][i] = toq()
    end
end

JLD.save("importance_sampling.jld", Dict("traces" => traces, "runtimes" => runtimes))
