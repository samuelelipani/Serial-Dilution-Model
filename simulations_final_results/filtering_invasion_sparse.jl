# this script gives, given the input of the species abundances for each dilution step, the check on whether a species different from 
# the most abundant one can invade the system. That means if it is able to survive wit a growth rate 
include("./find_steady_state_trial.jl") 
using DifferentialEquations, Distributions, Random, GLM, JLD2, FileIO

D = 100
tspanf = 12;
Ymag = ones(M)*100;
Ymag ./= sum(Ymag) 

indices = [];

for s in 1:5
    global n_load = FileIO.load("consumers_simulation1k_sparse_d$D.jld2","s")[s];
    global r_load = FileIO.load("resources_simulation1k_sparse_d$D.jld2","r")[s];
    global R_load = FileIO.load("consumption_matrix_simulation1k_sparse.jld2","R_save")[s];
    global N = size(n_load[end],2);
    global M = size(r_load[end],2);
    push!(indices,invasion(n_load,r_load,R_load,D,tspanf,Ymag,N,M))
end

FileIO.save("indices_sparse_d$D.jld2","indices",indices)
