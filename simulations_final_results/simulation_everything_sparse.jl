include("./find_steady_state_trial.jl") 
using DifferentialEquations, Distributions, Random, GLM, JLD2, FileIO

N = 200; 
M = 200;
Ymag = ones(M)*100;
Ymag ./= sum(Ymag);
si = [0.001,0.01,0.02,0.04,0.07,0.1,0.25,0.5];
steps = 300;
S = 0.1;
D = [1.05,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0];
tspanf = 8;


for d in D
    for h in 1:length(si)
        global s = []
        global r = []
        global R_save = []
        for iter in 1:10
            global sigma = si[h]
            global R = random_mat_uniform_sparse(N,M,si[h],0.1)
            global cons,ress = finds(N,M,R,d,Ymag,steps,tspanf);
            global ind_thresh = findall(cons[end][end,:] .>= 1e-9)
            cons = [cons[j][:,ind_thresh] for j in 1:2] # since conss is a vector with two components 
            global indices = invasion(cons,ress,R,d,tspanf,Ymag,size(cons[end],2),size(ress[end],2))
            cons = cons[end][:,indices]
            push!(s,cons)
            push!(r,ress)
            push!(R_save,R)
        end
        print("sigma $sigma done \n")
        FileIO.save("C:/Users/samue/Desktop/ecology/thesis/simulation_everything_2/consumers_simulation_$d"*"_s$sigma"*"_sparse.jld2","s",s)
        FileIO.save("C:/Users/samue/Desktop/ecology/thesis/simulation_everything_2/resources_simulation_$d"*"_s$sigma"*"_sparse.jld2","r",r)
        FileIO.save("C:/Users/samue/Desktop/ecology/thesis/simulation_everything_2/consumption_matrix_simulation_$d"*"_s$sigma"*"_sparse.jld2","R_save",R_save)
    end
    print("dilution $d done")
end