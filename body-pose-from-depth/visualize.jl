using FileIO
using Images: ImageCore

include("model.jl")
include("inference.jl")

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

Gen.load_generated_functions()

depth_renderer = BodyPoseRenderer(width, height, "localhost", 59894)
wireframe_renderer = BodyPoseWireframeRenderer(width, height, "localhost", 59895)

trace = simulate(generative_model, (depth_renderer,))
ground_truth = BodyPose(get_internal_node(get_choices(trace), :pose))
(original, blurred, observed) = get_call_record(trace).retval

#for n in [10, 100, 1000, 10000]
for n in [100000]
    println(n)
    inference = SIRPrior(depth_renderer, n)
    samples = BodyPose[]
    for i=1:5
        println("n=$n, replicate $i")
        push!(samples, infer(inference, observed))
    end
    visualize(wireframe_renderer, ground_truth, observed, samples, "vis-sir-prior-$n.png")
end
