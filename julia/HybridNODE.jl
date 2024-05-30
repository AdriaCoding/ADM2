using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random: Xoshiro; using CSV: read
using Plots, DataFrames
gr()

## Data retirieval
begin
rawdata = read("project2/datasets/Leigh1968_harelynx.csv", DataFrame)
df = mapcols(x -> Float32.(x .÷ 1000), rawdata[:,[:hare, :lynx]])
train_size = 20
df_train = df[1:train_size,:]
train_years = rawdata.year[1:train_size]

# Normalize data
scale = eachcol(df) .|> maximum |> transpose |> Array
const normalized_data = Array(df_train./scale)'
normalized_data' .* scale

#Display our data
dataplot = scatter(train_years, normalized_data[1,:], label="Hares", color="blue", lw=2)
scatter!(dataplot, train_years, normalized_data[2,:], label="Lynx", color="red", lw=2)
end

## Problem conditions
rng = Xoshiro(0)
u0 = normalized_data[:,1]
tspan = Float32.((0.0, train_size-1))
t = range(tspan[1], tspan[2], length=train_size) |> Array


## Define the network
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))

# Define the network 2->5->5->5->2
U = Chain(
    Dense(2,5,tanh), Dense(5,5, tanh), Dense(5,2)
    )
p_nn, state = Lux.setup(rng, U)
st = state

# WE FEED SOME GOOD INITIAL VALUES FOR THE PARAMETERS
p = ComponentArray(NN=p_nn, LV=[0.619f0, 1.986f0, 0.824f0, 2.40f0])

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    û = U(u, p.NN, st)[1] # Forward pass
    α, β, γ, δ = p.LV
    # Lokta-Volterra equations + ANN
    du[1] = α*u[1] - β*u[1]*u[2] + û[1]
    du[2] = γ*u[1]*u[2] - δ*u[2] + û[2]
end
prob_nn = ODEProblem(ude_dynamics!, u0, tspan, p)

function predict(θ; ODEalg = AutoTsit5(Rosenbrock23()), u0=u0, T = t)
    _prob = remake(prob_nn, u0 = u0 , tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, ODEalg, saveat = T,
    abstol = 1e-6, reltol = 1e-6,
    sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# Test the speed of several stiff ODE solvers.
#algs = [Rosenbrock23(), Tsit5(), Rodas5(), AutoTsit5(Rosenbrock23()),  AutoTsit5(Rodas4()), AutoTsit5(Rodas5())];
# To see reliable reults, run this line more than once.
#[@elapsed predict(p; ODEalg= alg) for alg in algs];
#[@elapsed predict(p; ODEalg= alg) for alg in algs]
#[@elapsed predict(p; ODEalg= alg) for alg in algs]

# Re-use the plot of the training data to paint new predicitons on top of it.
# Set up an increased time span to see the predictions more smoothly.
resolution = 8
t8 = range(tspan[1], tspan[2], length=train_size*resolution) |> Array
plot_years = range(rawdata.year[1], rawdata.year[train_size], length=train_size*resolution)
paint(p, plt = dataplot) = begin
    pred = predict(p; T=t8)
    predplot = plot(plt, plot_years, pred[1,:], label="Hares (pred)",
         color="blue", ylims=(-0.1,1.1))
    plot!(plot_years, pred[2,:], label="Lynx (pred)", color="red")
    display(predplot)
end

paint(p)

ξ = 5.0f0
function loss(p)
    pred = predict(p)
    return sum(abs2, normalized_data .- pred) + 
        + ξ * sum(abs2, p.NN) 
        #- 100.f0*min.(0.0f0, p.LV)
end


callback = function (opt_state, l; doplot=true)
    if opt_state.iter % 100 == 0
        println("Current loss after $(opt_state.iter) iterations: $(l)")
    end
    push!(losses, l)
    if opt_state.iter % 10 == 0
        if doplot
            p = opt_state.u
            paint(p)
        end
    end
    return false
end

# train this model!!

adtype = Optimization.AutoZygote();
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype);
optprob = Optimization.OptimizationProblem(optf, p);
losses = Float32[]
res1 = Optimization.solve(optprob, ADAM(0.01), callback = callback, maxiters = 500);

# Set final value for the trained parameters
plot(log10.(losses), label = nothing, xlabel="Iterations", ylabel="log-Loss")
p_trained = res1.u;

#Compare with test data 
begin
    test_data = df |> Array |> transpose;
    test_size = size(test_data, 2);
    t_test = range(tspan[2]+1, test_size-1 |> Float32, length=test_size-train_size)
    test_pred =  predict(p_trained; T=t_test, u0=normalized_data[:,end]);
    test_pred = test_pred .* scale'
    MSE = sum(abs2, test_pred .- test_data[:,train_size+1:end]) / (2*test_size)
    hares_MSE = sum(abs2, test_pred[1,:] .- test_data[1,train_size+1:end]) / (test_size)
    lynx_MSE = sum(abs2, test_pred[2,:] .- test_data[2,train_size+1:end]) / (test_size)
end 

println("MSE: ", MSE)
println("Hares MSE: $hares_MSE --> Avergage error: $(sqrt(hares_MSE))")
println("Lynx  MSE: $lynx_MSE --> Avergage error: $(sqrt(lynx_MSE))")
pp = round.(p_trained, digits=3);
println("α = $(pp[1]), β = $(pp[2]), δ = $(pp[3]), γ = $(pp[4])")
# Final Figure with results againts the test data
begin
    # First, plot the data 
    scatter(rawdata.year, test_data[1,:], label="Hares", lw=2)
    scatter!(rawdata.year, test_data[2,:], label="Lynx", lw=2)
    
    # Make new predictions 
    test_plot_t = Array(range(0.0f0, test_size-1 |> Float32, length=test_size*resolution))
    finalplot_years = test_plot_t .+ rawdata.year[1] 
    plot_trajectories = predict(p_trained; ODEalg=Tsit5(), T=test_plot_t) .* scale'
    sep = length(plot_years)
    plot_trajectories[1,1:sep]
    plot!(finalplot_years[1:sep], plot_trajectories[1,1:sep], label="Hares (NODE)", color="blue", lw=1)
    plot!(finalplot_years[1:sep], plot_trajectories[2,1:sep], label="Lynx (NODE)", color="red", lw=1)

    plot_trajectories[1,sep:end]
    finalplot_years[sep:end]
    plot!(finalplot_years[sep:end], plot_trajectories[1,sep:end], label=nothing, color="blue", lw=1, linestyle=:dash)
    plot!(finalplot_years[sep:end], plot_trajectories[2,sep:end], label=nothing, color="red", lw=1, linestyle=:dash)
    title!("Hares and Lynx population")
    xlabel!("Year")
    ylabel!("Population (in thousands)")
end


# EXTRAS
# Visualize the physics-informed model p_trained.LV = [α, β, δ, γ]
begin
    function lotka_volterra!(du, u, p, t)
        x, y = u
        α, β, δ, γ = p
        du[1] = dx = α * x - β * x * y
        du[2] = dy = -δ * y + γ * x * y
    end
    prob = ODEProblem(lotka_volterra!, u0, (0.0f0, 56.0f0), p_trained.LV)
    physics = solve(prob, Tsit5(), saveat = test_plot_t)
    hare_physics = physics[1,:] * scale[1]
    lynx_physics = physics[2,:] * scale[2]
    plot(finalplot_years, hare_physics, label="Hares (LV)", color="dodgerblue1", lw=1)
    plot!(finalplot_years, lynx_physics, label="Lynx (LV)", color="firebrick2", lw=1)
    scatter!(rawdata.year, test_data[1,:], label="Hares", lw=2)
    scatter!(rawdata.year, test_data[2,:], label="Lynx",  lw=2)
end

# Visualize the trained neural network as a 2D function

f(x, y) = begin
_x, _y = U(Float32.([x,y]), p_trained.NN, st)[1]
sqrt(_x^2 + _y^2)
end
surface(-1:0.01:1, -1:0.01:1, f, c=:viridis, xlabel="Hares", ylabel="Lynx", zlabel="|f(x,y)|")
f(u) = -f(u[1], u[2])
using Optim
optsol = optimize(f,[0.0, 0.0] )
println("Maximum value of ||NN|| in the unit square: $(-f(optsol.minimizer))")

# Baseline model?
μ = [sum(test_data[i,:])/57 for i in 1:2]; @show μ;
@show baseline_MSE = sum(abs2, μ.-test_data)/57;
@show baseline_average_error = sqrt(baseline_MSE);