# using DeepQLearning
# using POMDPs
using Flux
using BSON: @save, @load
using LinearAlgebra
using Zygote
# using POMDPModels
# using POMDPSimulators
# using POMDPTools
# using StatsBase;
# using TikzGraphs;
# using Printf;
# using GraphPlot;
# using Pkg;
using DataFrames; 
using CSV;
# using Graphs
# using StatsBase;
# using DelimitedFiles;

struct GradientQLearning
    𝒜  # action space (assumes 1:nactions)
    γ  # discount
    Q  # parameterized action value function Q(theta,s,a)
    ∇Q # gradient of action value function
    theta  # action value function parameter
    α  # learning rate
end

function lookahead(model::GradientQLearning, s, a)
    return model.Q(model.theta, s,a)
end

scale_gradient(∇, L2_max) = min(L2_max/norm(∇), 1)*∇

function update!(model::GradientQLearning, s, a, r, s′)
    𝒜, γ, Q, theta, α = model.𝒜, model.γ, model.Q, model.theta, model.α
    u = maximum(Q(theta,s′,a′) for a′ in 𝒜)
    for i in range(1, 6)
        myGradient = model.∇Q(theta,s,a)
        Δ = (r + γ*u - Q(theta,s,a)) .* myGradient[theta[i]]
        theta[i] .+= α*scale_gradient(Δ, 1)
    end
    return model
end

# define GradientQLearning
𝒜 = [[i, j] for j = 1:72 for i=1:54];
γ = 0.99;
# global model = Chain(
#     Dense(57, 32, Flux.relu),
#     Dense(32, 16, Flux.relu),
#     Dense(16, 1),
# )
@load "/home/thomas_ubuntu/Catan-AI-1/new_pass_largeset.bson" model
function Q(theta,s,a)
    params(model) = theta
    input = vcat(s, a)
    return sum(model(input))
end 
function ∇Q(theta,s,a)
    return gradient(()->sum(Q(theta,s,a)), theta)
end 
theta = Flux.params(model);
α = 0.01

myModel = GradientQLearning(𝒜, γ, Q, ∇Q, theta, α);
# s1 = [0 3 4 9 5 6 5 11 8 2 4 8 11 3 9 0 6 10 12 10 3 5 4 1 4 2 5 2 3 4 3 4 2 5 0 5 1 1 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
# a1 = [13 24]
# r1 = 0
# s2 = [0 3 4 9 5 6 5 11 8 2 4 8 11 3 9 0 6 10 12 10 3 5 4 1 4 2 5 2 3 4 3 4 2 5 0 5 1 1 3 12 -1 8 14 20 17 4 1 23 -1 16 27 36 29 9 4]
# a2 = [39 58]
# r2 = 0
# s3 = [0 3 4 9 5 6 5 11 8 2 4 8 11 3 9 0 6 10 12 10 3 5 4 1 4 2 5 2 3 4 3 4 2 5 0 5 1 1 3 12 38 8 14 20 17 4 1 23 57 16 27 36 29]
# a3 = [9 4] 
# r3 = 0
function myRead(filename)
    df = DataFrame(CSV.File(filename));
    D =  Matrix(df);
    return D, df;
end

D, df= myRead("/home/thomas_ubuntu/Catan-AI-1/code/data_medium.csv");
global S1 = D[:, 1:55]
global A1 = D[:, 56:57]
global R1 = [i*0 for i in range(1, size(D, 1))]
global S2 = D[:, 58:112] 
global A2 = D[:, 113:114]
global R2 = D[:, 170]
global S3 = D[:, 115:169]

for m in range(1, 10)
    # print(myModel.theta[1][1])
    # print("\n")
    for i in range(1, size(D, 1))
        s1 = S1[i, :];
        a1 = A1[i, :];
        r1 = R1[i];
        s2= S2[i, :];
        a2 = A2[i, :];
        r2 = R2[i];
        s3= S3[i, :];

        # update!(myModel, transpose(s1), transpose(a1), r1, transpose(s2));
        # update!(myModel, transpose(s2), transpose(a2), r2, transpose(s3));
        # print(r1)
        # print('\n')
        update!(myModel, s1, a1, r1, s2);
        # print(r2)
        # print('\n')
        update!(myModel, s2, a2, r2, s3);
        if mod(i,1000) == 0
            print((m, i))
            print("\n")
        end
        
    end
end


# @save "new_pass_largeset.bson" model







# myGradient = ∇Q(theta,transpose(s1),transpose(a1))

# sum(Q(theta, transpose(s1),transpose(a1)))
# global paramCount = 0
# for layer in model
#     global paramCount += sum(length, params(layer))
# end
# print(paramCount)