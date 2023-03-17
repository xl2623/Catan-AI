using Flux
using BSON: @save, @load
using LinearAlgebra
using Zygote
using DataFrames; 
using CSV;

@load "/home/thomas_ubuntu/Catan-AI-1/new_pass_largeset.bson" model

function Q(s,a)
    input = vcat(s, a)
    return sum(model(input))
end 

function decide(Q, 𝒜)
    a = argmax(Q, [a for a in 𝒜])
    return a
end

𝒜 = [[i, j] for j = 1:72 for i=1:54];

s1 = [0 3 4 9 5 6 5 11 8 2 4 8 11 3 9 0 6 10 12 10 3 5 4 1 4 2 5 2 3 4 3 4 2 5 0 5 1 1 3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
a1 = [13 24]
r1 = 0
s2 = [0 3 4 9 5 6 5 11 8 2 4 8 11 3 9 0 6 10 12 10 3 5 4 1 4 2 5 2 3 4 3 4 2 5 0 5 1 1 3 12 -1 8 14 20 17 4 1 23 -1 16 27 36 29 9 4]
a2 = [39 58]
r2 = 0
s3 = [0 3 4 9 5 6 5 11 8 2 4 8 11 3 9 0 6 10 12 10 3 5 4 1 4 2 5 2 3 4 3 4 2 5 0 5 1 1 3 12 38 8 14 20 17 4 1 23 57 16 27 36 29 9 4]
# a3 = [9 4] 
r3 = 0

Qa(a) = Q(transpose(s3),a)
# print(decide(Qa, 𝒜))
for i in range(1, 54)
    input = vcat(transpose(s1), [j for j in [i, 72]])
    print(i)
    print(": ")
    print(sum(model(input)))
    print("\n")
end