using FileIO
using Images: ImageCore

include("dynamic_model.jl")
include("dynamic_inference.jl")
include("neural_proposal.jl")

function trace_to_wireframe_movie(wireframe_renderer::BodyPoseWireframeRenderer, trace, trunk)
    retval = get_call_record(trace).retval
    num_steps = length(retval) + 1
    choices = get_choices(trace)
    poses = Vector{BodyPose}()
    poses[1] = BodyPose(get_internal_node(choices, :init_pose))
    step_retvals = get_call_record(trace).retval
    for (i, step_retval) in enumerate(step_retvals)
        poses[i + 1] = BodyPose(step_retval[1])
    end
    for (i, pose) in enumerate(poses)
        image = render(wireframe_renderer, pose)
        fname = @sprintf("%s-%03d.png", trunk, i)
        FileIO.save(fname, map(ImageCore.clamp01, image))
    end
end

function trace_to_observed_movie(depth_renderer::BodyPoseDepthRenderer, trace, trunk)
    retval = get_call_record(trace).retval
    num_steps = length(retval) + 1
    choices = get_choices(trace)
    init_image = choices[:init_image]
    fname = @sprintf("%s-%03d.png", trunk, 1)
    FileIO.save(fname, map(ImageCore.clamp01, init_image))
    step_retvals = get_call_record(trace).retval
    for (i, step_retval) in enumerate(step_retvals)
        image = step_retval[2]
        fname = @sprintf("%s-%03d.png", trunk, i + 1)
        FileIO.save(fname, map(ImageCore.clamp01, image))
    end
end

function poses_to_wireframe_movie(wireframe_renderer::BodyPoseWireframeRenderer, poses, trunk)
    for (i, pose) in enumerate(poses)
        image = render(wireframe_renderer, pose)
        fname = @sprintf("%s-%03d.png", trunk, i)
        FileIO.save(fname, map(ImageCore.clamp01, image))
    end
end

blender = "blender"
model = "HumanKTH.decimated.blend"
depth_renderer = BodyPoseDepthRenderer(width, height, blender, model, 59897)
wireframe_renderer = BodyPoseWireframeRenderer(width, height, blender, model, 59898)

# load large NN
arch = NetworkArchitecture(32, 32, 64, 1024)
proposal = make_neural_proposal(arch, neural_proposal_predict_beta)
session = init_session!(proposal.network)
params_fname = "params_arch_32_32_64_128-59902-36.jld"
as_default(GenTF.get_graph(proposal.network)) do
    saver = tf.train.Saver()
    tf.train.restore(saver, session, params_fname)
end

Gen.load_generated_functions()

# generate test sequence
println("generating test movie..")
import Random
Random.seed!(1)
trace = simulate(dynamic_generative_model, (50, depth_renderer))
retval = get_call_record(trace).retval
trace_to_observed_movie(depth_renderer, trace, "observed-movie")
choices = get_choices(trace)
#images = Vector{Matrix{Float64}}()
#push!(images, choices[:init_image])
#for i=1:49
    #push!(images, choices[:steps => i => :image])
#end
images = Matrix{Float64}[choices[:init_image] for i=1:50]

# run repeated importance sampling using deep net proposal with 10 particles
#println("running importance sampling on each frame..")
#movie_sir_nn = MovieSIRNN(depth_renderer, 100, proposal.neural_proposal)
#poses = infer(movie_sir_nn, images)
#poses_to_wireframe_movie(wireframe_renderer, poses, "movie-sir-nn-10")

# run particle filter
println("running particle filter..")
neural_pf = NeuralParticleFiltering(depth_renderer, 100, 50,
    proposal.init_neural_proposal, proposal.step_neural_proposal)
poses = infer(neural_pf, images)
poses_to_wireframe_movie(wireframe_renderer, poses, "movie-pf-nn-100")

#for n in [1, 10, 100]#, 1000]#, 10000]
#for n in [100000]
    #println(n)

    # MCMC
    #inference = MCMC(depth_renderer, n)
    #visualize(wireframe_renderer, ground_truth, observed, samples, "vis-mcmc-$n.png")

    # IS (prior)
    #inference = SIRPrior(depth_renderer, n)
    #visualize(wireframe_renderer, ground_truth, observed, samples, "vis-sir-prior-$n.png")

    # IS (NN)
    #inference = SIRNN(depth_renderer, n, proposal.neural_proposal)
    #samples = BodyPose[]
    #for i=1:4
        #println("n=$n, replicate $i")
        #push!(samples, infer(inference, observed))
    #end
    #visualize(wireframe_renderer, ground_truth, observed, samples, "vis-sir-nn-large-59902-$n.png")
#end

close(depth_renderer)
close(wireframe_renderer)
