using FileIO
using Images: ImageCore

include("model.jl")
include("inference.jl")
include("neural_proposal.jl")

function visualize(renderer::BodyPoseWireframeRenderer,
                   ground_truth::BodyPose, image::Matrix{Float64},
                   latent_samples::Vector{BodyPose}, fname::String)

    println(size(image))
    @assert size(image) == (height, width)

    # render wireframe for ground truth
    ground_truth_wireframe = render(renderer, ground_truth)
    println(size(ground_truth_wireframe))
    @assert size(ground_truth_wireframe) == (height, width)

    # render wireframes for latents
    latent_wireframes = Vector{Matrix{Float64}}()
    for pose in latent_samples
        img = render(renderer, pose)
        println(size(img))
        @assert size(img) == (height, width)
        push!(latent_wireframes, img)
    end
    
    # compose a bigger grayscale image
    combined = hcat(ground_truth_wireframe, image, latent_wireframes...)
    println(size(combined))
    @assert size(combined) == (height, (length(latent_samples) + 2) * width)

    # write to png
    FileIO.save(fname, map(ImageCore.clamp01, combined))
end


blender = "blender"
model = "HumanKTH.decimated.blend"
depth_renderer = BodyPoseDepthRenderer(width, height, blender, model, 59897)
wireframe_renderer = BodyPoseWireframeRenderer(width, height, blender, model, 59898)

# load large NN
arch = NetworkArchitecture(32, 32, 64, 1024)
proposal = make_neural_proposal(arch)
session = init_session!(proposal.network)
params_fname = "params_arch_32_32_64_128-59902-36.jld"
as_default(GenTF.get_graph(proposal.network)) do
    saver = tf.train.Saver()
    tf.train.restore(saver, session, params_fname)
end

Gen.load_generated_functions()

# generate test image
import Random
Random.seed!(1)
trace = simulate(generative_model, (depth_renderer,))
ground_truth = BodyPose(get_internal_node(get_choices(trace), :pose))
(original, blurred, observed) = get_call_record(trace).retval

for n in [1, 10, 100]#, 1000]#, 10000]
#for n in [100000]
    println(n)

    # MCMC
    #inference = MCMC(depth_renderer, n)
    #visualize(wireframe_renderer, ground_truth, observed, samples, "vis-mcmc-$n.png")

    # IS (prior)
    #inference = SIRPrior(depth_renderer, n)
    #visualize(wireframe_renderer, ground_truth, observed, samples, "vis-sir-prior-$n.png")

    # IS (NN)
    inference = SIRNN(depth_renderer, n, proposal.neural_proposal)
    samples = BodyPose[]
    for i=1:4
        println("n=$n, replicate $i")
        push!(samples, infer(inference, observed))
    end
    visualize(wireframe_renderer, ground_truth, observed, samples, "vis-sir-nn-large-59902-$n.png")
end

close(depth_renderer)
close(wireframe_renderer)
