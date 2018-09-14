using DataFrames
using CSV
using PyPlot
using Query
using DataValues: DataValue

df = CSV.read("sir-prior.csv")
println(df.elapsed)
println(df.square_error)

Base.middle(a::DataValue{Float64}, b::DataValue{Float64}) = DataValue(middle(get(a), get(b)))

summarized = @from row in df begin
    @group row by row.key into g
    @select {name=key(g), median_elapsed=mean(g.elapsed), rmse=sqrt(mean(g.square_error))}
    @collect DataFrame
end

println(summarized)


#scatter(df[:elapsed], df[:square_error])

xlabel("elapsed")
ylabel("RMSE")
savefig("plot.png")
