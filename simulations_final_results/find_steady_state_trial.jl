function CR!(dx,x,p,t)
    N,M,R = p
    dx[1:N] = x[1:N].*(R*x[1+N:N+M])
    dx[N+1:N+M] = -x[1+N:N+M].*(transpose(R)*x[1:N])   
end

function random_mat_uniform(N,M,σ)
    a = rand(Uniform(1-σ,1+σ),N,M);
    return a 
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

function finds(N,M,R,D,Ymag,steps,tspanf)

    X0s = Array{Any}(undef,steps+1);
    Y0s = Array{Any}(undef,steps+1);
    sol_mat = Matrix{Float64}[];
    res_mat = Matrix{Float64}[];
    X0s[1] = ones(N)/N*(D-1);
    X0s[1] ./= D
    Y0s[1] = Ymag

    p = [N,M,R]
    tspan = (0.,tspanf)

    for j in 1:steps

        X0 = X0s[j]
        Y0 = Y0s[j]
        XY0 = vcat(X0,Y0)
        prob = ODEProblem(CR!, XY0, tspan, p)
        sol_tmp = solve(prob,Tsit5(),abstol=1e-12,reltol=1e-12)

        X0s[j+1] = sol_tmp.u[length(sol_tmp.u)][1:p[1]]./D
        Y0s[j+1] = sol_tmp.u[length(sol_tmp.u)][(p[1]+1):(p[1]+M)]./D .+ Y0s[1]
        
        sol = [sol_tmp.u[i][1:p[1]] for i in 1:length(sol_tmp.u)]   
        res = [sol_tmp.u[i][(p[1]+1):(p[1]+M)] for i in 1:length(sol_tmp.u)]

        if j in [steps-1,steps]
            tmp = zeros(length(sol),p[1])
            for i in 1:length(sol)
                tmp[i,:] = sol[i]
            end
            push!(sol_mat,tmp)

            tmp_res = zeros(length(res),M)
            for i in 1:length(res)
                tmp_res[i,:] = res[i]
            end
            push!(res_mat,tmp_res)
        end
        
        print("dilution step number $j done\n")
    end

    return sol_mat,res_mat

end

function invasion(n_load,r_load,R_load,D,tspanf,Ymag,N,M)
    
    decrease_rates = Matrix{Float64}(undef,1,N);
    decrease_rates[1,:] = log.((n_load[end][end,:])) - log.(n_load[end-1][end,:])

    n_ex = n_load[end];
    positive_decrease_rates = findall(decrease_rates[end,:] .>= 0);
    negative_decrease_rates = findall(decrease_rates[end,:] .< 0);
    if length(negative_decrease_rates) != 0
        first_negative_decrease_rates_index = findall(n_ex[end,:] .== sort(n_ex[end,negative_decrease_rates],rev=true)[1]);
        good_indices = vcat(positive_decrease_rates,first_negative_decrease_rates_index);

        N1 = length(good_indices); 
        new_R_load = vcat(R_load[positive_decrease_rates,:],R_load[first_negative_decrease_rates_index,:])
        tspan = (0.,tspanf)
        sol_mat = n_ex[:,good_indices];

        p = [N1,M,new_R_load];

        # INVASION ANALYSIS 
        X0 = [sol_mat[end,1:N1-1]/D...,1e-20]; 
        # i'm changing the abundance of the second most abundant species in the environment to a very small one. 
        # we want to see whether the latter is gonna manage to invade the system, that is if its growth rate after one step of evolution is greater or less than D
        Y0 = r_load[end][end,:]./D + Ymag 
        XY0 = vcat(X0,Y0);

        prob = ODEProblem(CR!, XY0, tspan, p)
        sol_tmp = solve(prob,Tsit5(),abstol=1e-8,reltol=1e-8)
        sol = [sol_tmp.u[i][1:p[1]] for i in 1:length(sol_tmp.u)];
        res = [sol_tmp.u[i][(p[1]+1):(p[1]+M)] for i in 1:length(sol_tmp.u)];
        sol_mat = zeros(length(sol),p[1])
        for i in 1:length(sol)
            sol_mat[i,:] = sol[i]
        end

        inv = ones(length(positive_decrease_rates));
        n = sum(inv);
        if sol_mat[end,end]./sol_mat[1,end] > D 
            push!(inv,1);
        else
            push!(inv,0);
        end 
        print("$n species survived and first invasion done \n")

        # IF INVASION HAPPENS, SPECIES GROWTH RATE IS GREATER THAN D, MAYBE THERE IS THE POSSIBILITY THAT ALSO THE SPECIES WHOSE ABUNDANCE IS 
        # SMALLER THAN THE PREVIOUS ONE AND HAS A NEGATIVE DECREASE RATE CAN INVADE. WE CAN REPEAT THE PROCESS UNTIL ALL THE SPECIES INVADE OR 
        # WE FIND ONE THAT IS NOT ABLE TO GET INTO THE SYSTEM

        di = sort(Dict(n_ex[end,negative_decrease_rates] .=> negative_decrease_rates), by = x -> x[1], rev=true)

        # from now on we should introduce into the poll of species those that are into the positive_decrease_rates_indices and each of those that are into the ordered
        # dictionaty. It represents species whose decrease rate is negative and ordered in terms of their abundances. Including them into the poll of the above 
        # mentioned species should start from the second element of the dictionary when the first is able to INVADE the system

        negative_decrease_rates_ordered = [values(di)...][2:end];

        for i in 1:(length(negative_decrease_rates_ordered))
            push!(good_indices,negative_decrease_rates_ordered[i])

            N1 = length(good_indices); 
            new_R_load = R_load[good_indices,:]
            tspan = (0.,tspanf)
            sol_mat = n_ex[:,good_indices];
            p = [N1,M,new_R_load];
            X0 = [sol_mat[end,1:N1-1]/D...,1e-20]; 
            Y0 = r_load[end][end,:]./D + Ymag 
            XY0 = vcat(X0,Y0);

            prob = ODEProblem(CR!, XY0, tspan, p)
            sol_tmp = solve(prob,Tsit5(),abstol=1e-8,reltol=1e-8)
            sol = [sol_tmp.u[i][1:p[1]] for i in 1:length(sol_tmp.u)];
            res = [sol_tmp.u[i][(p[1]+1):(p[1]+M)] for i in 1:length(sol_tmp.u)];
            sol_mat = zeros(length(sol),p[1])
            for i in 1:length(sol)
                sol_mat[i,:] = sol[i]
            end

            if sol_mat[end,end]./sol_mat[1,end] > D 
                push!(inv,1)
            else
                push!(inv,0)
                break
            end 
            step = sum(inv)
            print("invasion $step done \n")
        end

        invaders = sum(inv);

        if invaders == length(inv) 
            print("All species survive \n")
        else
            print("$invaders species survived and those with positive decrease rates were $n \n")
        end
    else 
        good_indices = positive_decrease_rates
        invaders = length(positive_decrease_rates)
        print("$invaders species survived and there were no species with negative decrease rate \n")
    end
    return good_indices
end