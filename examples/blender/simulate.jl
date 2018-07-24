include("model.jl")

import JLD

traces = Trace[]
ground_truth_images = Matrix{Float64}[]
blurred_images = Matrix{Float64}[]
observable_images = Matrix{Float64}[]
for i=1:100
    println("simulation $i")
    (trace, _, (ground_truth_image, blurred_image, observable_image)) = simulate(model, ())
    push!(ground_truth_images, ground_truth_image)
    push!(blurred_images, blurred_image)
    push!(observable_images, observable_image)
    push!(traces, trace)
    output_filename = @sprintf("simulated.observable.%03d.png", i)
    FileIO.save(output_filename, map(ImageCore.clamp01, observable_image))
    output_filename = @sprintf("simulated.ground_truth.%03d.png", i)
    FileIO.save(output_filename, map(ImageCore.clamp01, ground_truth_image))
end
JLD.save("simulation_data.jld", Dict(
    "traces" => traces,
    "ground_truth_images" => ground_truth_images,
    "blurred_images" => blurred_images,
    "observable_images" => observable_images))
