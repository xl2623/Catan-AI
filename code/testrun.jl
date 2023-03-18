using Flux
using BSON: @save, @load
using LinearAlgebra
using Zygote
using DataFrames; 
using CSV;


struct GradientQLearning
    A 
    gamma 
    Q
    gradQ
    theta 
    alpha 
end

function lookahead(model::GradientQLearning, s, a)
    return model.Q(model.theta, s,a)
end

scale_gradient(∇, L2_max) = min(L2_max/norm(∇), 1)*∇

function update!(model::GradientQLearning, s, a, r, sp, ignore_expected_util)
    A, gamma, Q, theta, alpha = model.A, model.gamma, model.Q, model.theta, model.alpha
    if !ignore_expected_util
        u = maximum(Q(theta,sp,ap) for ap in A)
    else
        u = 0
    end
    for i in range(1, 6)
        myGradient = model.gradQ(theta,s,a)
        Δ = (r + gamma*u - Q(theta,s,a)) .* myGradient[theta[i]]
        theta[i] .+= alpha*scale_gradient(Δ, 1)
    end
    return model
end

global basis = Chain(
    Dense(57, 32, Flux.relu),
    Dense(32, 16, Flux.relu),
    Dense(16, 1),
)
# @load "test1.bson" basis
function Q(theta,s,a)
    params(basis) = theta
    input = vcat(s, a)
    return sum(basis(input))
end 

function gradQ(theta,s,a)
    return gradient(()->sum(Q(theta,s,a)), theta)
end 

function myRead(filename)
    df = DataFrame(CSV.File(filename));
    D =  Matrix(df);
    return D, df;
end

function train(dataFile, saveFile)
    # define GradientQLearning
    A = [[i, j] for j = 1:72 for i=1:54];
    gamma = 0.99;
    theta = Flux.params(basis);
    alpha = 0.0005

    myModel = GradientQLearning(A, gamma, Q, gradQ, theta, alpha);

    D, df= myRead(dataFile);

    S1 = D[:, 1:55]
    A1 = D[:, 56:57]
    R1 = [i*0 for i in range(1, size(D, 1))]
    S2 = D[:, 58:112] 
    A2 = D[:, 113:114]
    R2 = D[:, 170]
    S3 = D[:, 115:169]

    for m in range(1, 10)
        print(theta[1][1])
        print("\n")
        for i in range(1, size(D, 1))
            s1 = S1[i, :];
            a1 = A1[i, :];
            r1 = R1[i];
            s2= S2[i, :];
            a2 = A2[i, :];
            r2 = R2[i];
            s3= S3[i, :];

            update!(myModel, s1, a1, r1, s2, false);
            update!(myModel, s2, a2, r2, s3, true);

            if mod(i,1000) == 0
                print((m, i))
                print("\n")
            end
        end
        theta = Flux.params(basis);
        print(theta[1][1])
        print("\n")
    end
    @save saveFile basis
end

train("/home/thomas_ubuntu/Catan-AI-1/code/data/data.csv", "alpha_5e-4_gamma_99e-2_epoch_10.bson")

