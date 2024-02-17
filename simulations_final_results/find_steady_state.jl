function CR!(dx,x,p,t)
    N,M,R = p
    dx[1:N] = x[1:N].*(R*x[1+N:N+M])
    dx[N+1:N+M] = -x[1+N:N+M].*(transpose(R)*x[1:N])   
end

function random_mat(N,M,σ)
    a = rand(Normal(1,σ*sqrt(M)),N,M);
    return a
end

# function random_mat_uniform(N,M,σ)
#     a = rand(Uniform(1-sqrt(12*M)*σ,1+sqrt(12*M)*σ),N,M);
#     return a 
# end

function random_mat_uniform(N,M,σ)
    a = rand(Uniform(1-σ,1+σ),N,M);
    return a 
end

function finds(N,M,R,D,Ymag,steps,ab_thresh,steady_threshold,tspanf)
    X0s = Array{Any}(undef,steps+1);
    Y0s = Array{Any}(undef,steps+1);
    Xfs_dummy = Array{Any}(undef,steps);
    sol_mat = Matrix{Float64}[];
    res_mat = Matrix{Float64}[];
    time_series = Array{Float64}[];
    X0s[1] = ones(N)/N*100;
    X0s[1] ./= D
    Y0s[1] = Ymag

    p = [N,M,R]
    tspan = (0.,tspanf)
    flag = 0 
    steady_steps = 0
    no_neg_flag = 0

    for j in 1:steps

        X0 = X0s[j]
        Y0 = Y0s[j]
        XY0 = vcat(X0,Y0)
        prob = ODEProblem(CR!, XY0, tspan, p)
        sol_tmp = solve(prob,Tsit5(),abstol=1e-9,reltol=1e-8)

        Xfs_dummy[j] = sol_tmp.u[length(sol_tmp.u)][1:p[1]]
        X0s[j+1] = sol_tmp.u[length(sol_tmp.u)][1:p[1]]./D
        Y0s[j+1] = sol_tmp.u[length(sol_tmp.u)][(p[1]+1):(p[1]+M)]./D .+ Ymag
        
        # bad_index_x = findall((sol_tmp.u[length(sol_tmp.u)][1:p[1]]) .< ab_thresh)
        bad_index_x = findall(X0s[j+1] .< ab_thresh)

        sol = [sol_tmp.u[i][1:p[1]] for i in 1:length(sol_tmp.u)]
        res = [sol_tmp.u[i][(p[1]+1):(p[1]+M)] for i in 1:length(sol_tmp.u)]
        # push!(time_series,sol_tmp.t)
        time_series = sol_tmp.t
        
        tmp = zeros(length(sol),p[1])
        for i in 1:length(sol)
            tmp[i,:] = sol[i]
        end
        # push!(sol_mat,tmp[:,[i for i in 1: length(X0s[j+1]) if i ∉ bad_index_x]])
        sol_mat = tmp[:,[i for i in 1: length(X0s[j+1]) if i ∉ bad_index_x]]
 
        tmp_res = zeros(length(res),M)
        for i in 1:length(res)
            tmp_res[i,:] = res[i]
        end
        # push!(res_mat,tmp_res)
        res_mat = tmp_res

        if sum(tmp_res .< 0) == 0
            no_neg_flag += 1
        end

        X0s[j+1] = X0s[j+1][[i for i in 1: length(X0s[j+1]) if i ∉ bad_index_x]]
        Xfs_dummy[j] = Xfs_dummy[j][[i for i in 1: length(Xfs_dummy[j]) if i ∉ bad_index_x]]
        R = R[[i for i in 1: length(X0s[j]) if i ∉ bad_index_x],:]
        p = [length(X0s[j+1]),M,R]
        
        X0_next_dummy = Xfs_dummy[j]
        X0_now_dummy = X0s[j]
        X0_now_dummy = X0_now_dummy[[i for i in 1: length(X0_now_dummy) if i ∉ bad_index_x]]
        
        steady_steps += 1
        if ((mean(abs.((X0_next_dummy./X0_now_dummy) .- D)) .< D*steady_threshold) && flag == 0) || p[1] == 0
            flag = 1 
            break
        end
        print("$steady_steps dilution step done \n")
    end

    if steady_steps == no_neg_flag 
        print("No negative resources for the whole process \n")
    end

    return [sol_mat,res_mat,R,steady_steps],time_series

end