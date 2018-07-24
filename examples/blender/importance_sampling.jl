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

function do_inference(input_image, n::Int)

    # construct trace containing observed image
    observations = Trace()
    observations["image"] = input_image

    # do importance sampling
    traces = Vector{Trace}(n)
    log_weights = Vector{Float64}(n)
    for i=1:n
        (traces[i], _, log_weights[i], _) = Gen.imp2(model, (), dl_proposal, (), observations)
    end

    dist = exp.(log_weights - logsumexp(log_weights))
    idx = rand(categorical, (dist,))
    traces[idx]
end

tf.run(get_ambient_tf_session(), tf.global_variables_initializer())
saver = tf.train.Saver()
tf.train.restore(saver, get_ambient_tf_session(), "inference_network_params.jld")

input_fname = "images/simulated.observable.086.png"
input_image = convert(Matrix{Float64}, FileIO.load(input_fname))
for n in [10, 100, 1000]
    reps = 20
    runtimes = Vector{Float64}(reps)
    for i=1:reps
        tic()
        predicted = do_inference(input_image, n)
        runtimes[i] = toq()
        output_fname = @sprintf("importance-sampling/n%03d.%03d.png", n, i)
        render_trace(predicted, output_fname)
    end
    println("n: $n, median runtime (sec.): $(median(runtimes))")
end
