import JLD

include("model.jl")

## render ground truth wireframe
#println("rendering ground truths...")
#for (i, trace) in enumerate(JLD.load("simulation_data.jld")["traces"])
    #set_body_pose!(client, trace_to_pose(trace))
    #setup_for_wireframe!(client)
    #set_resolution!(client, 400, 400)
    #render(client, @sprintf("ground_truth/ground_truth.%03d.png", i))
    #setup_for_depth!(client)
    #render(client, @sprintf("ground_truth/ground_truth.%03d.depth.png", i))
#end
#
## render prior sample wireframes
#println("rendering prior samples...")
#for i=1:10
    #(trace, _, _) = simulate(model, ())
    #set_body_pose!(client, trace_to_pose(trace))
    #setup_for_wireframe!(client)
    #set_resolution!(client, 400, 400)
    #render(client, @sprintf("prior/%03d.png", i))
    #setup_for_depth!(client)
    #render(client, @sprintf("prior/%03d.depth.png", i))
#end

#exit()

using GenLiteTF
using TensorFlow
tf = TensorFlow

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
    latent_images = Vector{Matrix{Float64}}(n)
    traces = Vector{Trace}(n)
    log_weights = Vector{Float64}(n)
    for i=1:n
        (traces[i], _, log_weights[i], (latent_images[i], _, _)) = Gen.imp2(
            model, (), dl_proposal, (), observations)
    end

    dist = exp.(log_weights - logsumexp(log_weights))
    idx = rand(categorical, (dist,))
    (traces[idx], latent_images[idx])
end

tf.run(get_ambient_tf_session(), tf.global_variables_initializer())
saver = tf.train.Saver()
tf.train.restore(saver, get_ambient_tf_session(), "inference_network_params.jld")

input_fname = "simulated.observable.009.png"
input_image = convert(Matrix{Float64}, FileIO.load(input_fname))
for n in [1, 10, 100, 1000]
    println("importance sampling n=$n")
    reps = 20
    runtimes = Vector{Float64}(reps)
    for i=1:reps
        tic()

        # do inference
        setup_for_depth!(client)
        set_resolution!(client, width, height)
        (predicted, _) = do_inference(input_image, n)
        runtimes[i] = toq()

        # render the inferred wireframe
        pose = trace_to_pose(predicted)
        set_body_pose!(client, pose)
        setup_for_wireframe!(client)
        set_resolution!(client, 400, 400)
        render(client, @sprintf("importance-sampling/n%03d.%03d.png", n, i))

        # render the inferred depth image
        setup_for_depth!(client)
        render(client, @sprintf("importance-sampling/n%03d.%03d.depth.png", n, i))

    end
    println("n: $n, median runtime (sec.): $(median(runtimes))")
end
