using Plots, DifferentialEquations, Distributions, Random, LaTeXStrings, GLM, FileIO

function CR!(dx,x,p,t)
    N,M,R = p
    dx[1:N] = x[1:N].*(R*x[1+N:N+M])
    dx[N+1:N+M] = -x[1+N:N+M].*(transpose(R)*x[1:N])   
end
 
function random_mat_uniform_sparse(N,M,σ,S)
    a = rand(Uniform(1-σ,1+σ),N,M);
    for i in 1:N, j in 1:M
        r = rand()
        if r < S
            global a[i,j] = 0
        end
    end
    return a 
end

function finds(N,M,R,D,Ymag,steps,ab_thresh,steady_threshold,tspanf)

    X0s = Array{Any}(undef,steps+1);
    Xfs_dummy = Array{Any}(undef,steps);
    Y0s = Array{Any}(undef,steps+1);
    sol_mat = Matrix{Float64}[];
    X0s[1] = ones(N)/N*(D-1);
    X0s[1] ./= D
    Y0s[1] = Ymag

    p = [N,M,R]
    tspan = (0.,tspanf)
    flag = 0 
    steady_steps = 0

    for j in 1:steps

        X0 = X0s[j]
        Y0 = Y0s[j]
        XY0 = vcat(X0,Y0)
        prob = ODEProblem(CR!, XY0, tspan, p)
        sol_tmp = solve(prob,Tsit5(),abstol=1e-12,reltol=1e-12)

        Xfs_dummy[j] = sol_tmp.u[length(sol_tmp.u)][1:p[1]]
        X0s[j+1] = sol_tmp.u[length(sol_tmp.u)][1:p[1]]./D
        Y0s[j+1] = sol_tmp.u[length(sol_tmp.u)][(p[1]+1):(p[1]+M)]./D .+ Y0s[1]
        
        bad_index_x = findall((sol_tmp.u[length(sol_tmp.u)][1:p[1]]) .< ab_thresh)

        sol = [sol_tmp.u[i][1:p[1]] for i in 1:length(sol_tmp.u)]        
        tmp = zeros(length(sol),p[1])
        for i in 1:length(sol)
            tmp[i,:] = sol[i]
        end
        sol_mat = tmp[:,[i for i in 1: length(X0s[j+1]) if i ∉ bad_index_x]]

        X0s[j+1] = X0s[j+1][[i for i in 1: length(X0s[j+1]) if i ∉ bad_index_x]]
        Xfs_dummy[j] = Xfs_dummy[j][[i for i in 1: length(Xfs_dummy[j]) if i ∉ bad_index_x]]
        R = R[[i for i in 1: length(X0s[j]) if i ∉ bad_index_x],:]
        p = [length(X0s[j+1]),M,R]
         
        X0_next_dummy = Xfs_dummy[j]
        X0_now_dummy = X0s[j]
        X0_now_dummy = X0_now_dummy[[i for i in 1: length(X0_now_dummy) if i ∉ bad_index_x]]
        
        steady_steps += 1
        # print("dilution step number $steady_steps done\n")
        if ((mean(abs.((X0_next_dummy./X0_now_dummy) .- D)) .< D*steady_threshold) && flag == 0) || p[1] == 0
            flag = 1 
            break
        end
    end

    return sol_mat

end

N = 50; 
M = 50;
S = 0.1;
Ymag = ones(M)*1000;
Ymag ./= sum(Ymag) 
si = [0.0001,0.001,0.01,0.1,0.25,0.5];
# tspanf = 0.05
tspanf = collect(4:1:15);
steps = 500;
steady_threshold = 1e-2;
ab_thresh = 1e-2;
D = collect(20:2:100);

reals = 100
u = [fill(NaN,size(D, 1),size(tspanf, 1)) for i in 1:reals];
u_fin = fill(NaN,size(D, 1),size(tspanf, 1));

for h in 1:length(si)
    sigma = si[h]
    for k in 1:reals
        global R = random_mat_uniform_sparse(N,M,si[h],S)
        for i in eachindex(D), j in eachindex(tspanf)
            D1 = D[i];
            tspanf1 = tspanf[j]
            global s = finds(N,M,R,D1,Ymag,steps,ab_thresh,steady_threshold,tspanf1);
            global u[k][i, j] = size(s,2);
            print("inner iteration for $D1 and $tspanf1 done \n")
        end
        print("Iteration $k out of $reals done \n")
        global u_fin = (sum(u)/reals);
        FileIO.save("heatmap_simulation_s$sigma.jld2","u_fin",u_fin)
    end
end