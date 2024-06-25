### sdp relaxations in the rectangular W-space
import LinearAlgebra: Hermitian, cholesky, Symmetric, diag, I
import SparseArrays: SparseMatrixCSC, sparse, spdiagm, findnz, spzeros, nonzeros

""
function constraint_current_limit_from(pm::AbstractWRMModel, n::Int, f_idx, c_rating_a)
    l,i,j = f_idx

    w_fr = var(pm, n, :w, i)

    p_fr = var(pm, n, :p, f_idx)
    q_fr = var(pm, n, :q, f_idx)
    JuMP.@constraint(pm.model, [w_fr*c_rating_a^2+1, 2*p_fr, 2*q_fr, w_fr*c_rating_a^2-1] in JuMP.SecondOrderCone())
end

""
function constraint_current_limit_to(pm::AbstractWRMModel, n::Int, t_idx, c_rating_a)
    l,j,i = t_idx

    w_to = var(pm, n, :w, j)

    p_to = var(pm, n, :p, t_idx)
    q_to = var(pm, n, :q, t_idx)
    JuMP.@constraint(pm.model, [w_to*c_rating_a^2+1, 2*p_to, 2*q_to, w_to*c_rating_a^2-1] in JuMP.SecondOrderCone())
end




""
function constraint_model_voltage(pm::AbstractWRMModel, n::Int)
    _check_missing_keys(var(pm, n), [:WR,:WI], typeof(pm))

    WR = var(pm, n)[:WR]
    WI = var(pm, n)[:WI]

    JuMP.@constraint(pm.model, [WR WI; -WI WR] in JuMP.PSDCone())
end


""
function variable_bus_voltage(pm::AbstractWRMModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    wr_min, wr_max, wi_min, wi_max = ref_calc_voltage_product_bounds(ref(pm, nw, :buspairs))
    bus_ids = ids(pm, nw, :bus)

    w_index = 1:length(bus_ids)
    lookup_w_index = Dict((bi,i) for (i,bi) in enumerate(bus_ids))

    WR_start = zeros(length(bus_ids), length(bus_ids)) + I

    WR = var(pm, nw)[:WR] = JuMP.@variable(pm.model,
        [i=1:length(bus_ids), j=1:length(bus_ids)], Symmetric, base_name="$(nw)_WR", start=WR_start[i,j]
    )
    if report
        sol(pm, nw)[:WR] = WR
    end

    WI = var(pm, nw)[:WI] = JuMP.@variable(pm.model,
        [1:length(bus_ids), 1:length(bus_ids)], base_name="$(nw)_WI", start=0.0
    )
    if report
        sol(pm, nw)[:WI] = WI
    end

    # bounds on diagonal
    for (i, bus) in ref(pm, nw, :bus)
        w_idx = lookup_w_index[i]
        wr_ii = WR[w_idx,w_idx]
        wi_ii = WR[w_idx,w_idx]

        if bounded
            JuMP.set_lower_bound(wr_ii, (bus["vmin"])^2)
            JuMP.set_upper_bound(wr_ii, (bus["vmax"])^2)

            #this breaks SCS on the 3 bus exmple
            #JuMP.set_lower_bound(wi_ii, 0)
            #JuMP.set_upper_bound(wi_ii, 0)
        else
            JuMP.set_lower_bound(wr_ii, 0)
        end
    end

    # bounds on off-diagonal
    for (i,j) in ids(pm, nw, :buspairs)
        wi_idx = lookup_w_index[i]
        wj_idx = lookup_w_index[j]

        if bounded
            JuMP.set_upper_bound(WR[wi_idx, wj_idx], wr_max[(i,j)])
            JuMP.set_lower_bound(WR[wi_idx, wj_idx], wr_min[(i,j)])

            JuMP.set_upper_bound(WI[wi_idx, wj_idx], wi_max[(i,j)])
            JuMP.set_lower_bound(WI[wi_idx, wj_idx], wi_min[(i,j)])
        end
    end

    var(pm, nw)[:w] = Dict{Int,Any}()
    for (i, bus) in ref(pm, nw, :bus)
        w_idx = lookup_w_index[i]
        var(pm, nw, :w)[i] = WR[w_idx,w_idx]
    end
    report && sol_component_value(pm, nw, :bus, :w, ids(pm, nw, :bus), var(pm, nw)[:w])

    var(pm, nw)[:wr] = Dict{Tuple{Int,Int},Any}()
    var(pm, nw)[:wi] = Dict{Tuple{Int,Int},Any}()
    for (i,j) in ids(pm, nw, :buspairs)
        w_fr_index = lookup_w_index[i]
        w_to_index = lookup_w_index[j]

        var(pm, nw, :wr)[(i,j)] = WR[w_fr_index, w_to_index]
        var(pm, nw, :wi)[(i,j)] = WI[w_fr_index, w_to_index]
    end
    report && sol_component_value_buspair(pm, nw, :buspairs, :wr, ids(pm, nw, :buspairs), var(pm, nw)[:wr])
    report && sol_component_value_buspair(pm, nw, :buspairs, :wi, ids(pm, nw, :buspairs), var(pm, nw)[:wi])
end




###### Sparse SDP Relaxations ######


struct _SDconstraintDecomposition
    "Each sub-vector consists of bus IDs corresponding to a clique grouping"
    decomp::Vector{Vector{Int}}
    "`lookup_index[bus_id] --> idx` for mapping between 1:n and bus indices"
    lookup_index::Dict
    "A chordal extension and maximal cliques are uniquely determined by a graph ordering"
    ordering::Vector{Int}
end
import Base: ==
function ==(d1::_SDconstraintDecomposition, d2::_SDconstraintDecomposition)
    eq = true
    for f in fieldnames(_SDconstraintDecomposition)
        eq = eq && (getfield(d1, f) == getfield(d2, f))
    end
    return eq
end

function variable_bus_voltage(pm::AbstractSparseSDPWRMModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)

    if haskey(pm.ext, :SDconstraintDecomposition)
        decomp = pm.ext[:SDconstraintDecomposition]
        groups = decomp.decomp
        lookup_index = decomp.lookup_index
        lookup_bus_index = Dict((reverse(p) for p = pairs(lookup_index)))
    else
        cadj, lookup_index, ordering = _chordal_extension(pm, nw)
        groups = _maximal_cliques(cadj)
        lookup_bus_index = Dict((reverse(p) for p = pairs(lookup_index)))
        groups = [[lookup_bus_index[gi] for gi in g] for g in groups]
        pm.ext[:SDconstraintDecomposition] = _SDconstraintDecomposition(groups, lookup_index, ordering)
    end

    voltage_product_groups =
        var(pm, nw)[:voltage_product_groups] =
        Vector{Dict{Symbol, Array{JuMP.VariableRef,2}}}(undef, length(groups))

    for (gidx, group) in enumerate(groups)
        n = length(group)
        wr_start = zeros(n, n) + I
        voltage_product_groups[gidx] = Dict()
        WR = voltage_product_groups[gidx][:WR] =
            var(pm, nw)[:voltage_product_groups][gidx][:WR] =
            JuMP.@variable(pm.model, [i=1:n, j=1:n], Symmetric,
                base_name="$(nw)_$(gidx)_WR", start=wr_start[i,j])
        if report
            sol(pm, nw, :w_group, gidx)[:WR] = WR
        end

        WI = voltage_product_groups[gidx][:WI] =
            var(pm, nw)[:voltage_product_groups][gidx][:WI] =
            JuMP.@variable(pm.model, [1:n, 1:n],
                base_name="$(nw)_$(gidx)_WI", start=0.0)
        if report
            sol(pm, nw, :w_group, gidx)[:WI] = WI
        end
    end

    # voltage product bounds
    visited_buses = []
    visited_buspairs = []
    var(pm, nw)[:w] = Dict{Int,Any}()
    var(pm, nw)[:wr] = Dict{Tuple{Int,Int},Any}()
    var(pm, nw)[:wi] = Dict{Tuple{Int,Int},Any}()
    wr_min, wr_max, wi_min, wi_max = ref_calc_voltage_product_bounds(ref(pm, nw, :buspairs))
    for (gidx, voltage_product_group) in enumerate(voltage_product_groups)
        WR, WI = voltage_product_group[:WR], voltage_product_group[:WI]
        group = groups[gidx]
        ng = length(group)

        # diagonal bounds
        for (group_idx, bus_id) in enumerate(group)
            # group_idx indexes into group
            # bus_id indexes into ref(pm, nw, :bus)
            bus = ref(pm, nw, :bus, bus_id)

            wr_ii = WR[group_idx, group_idx]

            if bounded
                JuMP.set_upper_bound(wr_ii, (bus["vmax"])^2)
                JuMP.set_lower_bound(wr_ii, (bus["vmin"])^2)
            else
                JuMP.set_lower_bound(wr_ii, 0)
            end

            # for non-semidefinite constraints
            if !(bus_id in visited_buses)
                push!(visited_buses, bus_id)
                var(pm, nw, :w)[bus_id] = wr_ii
            end
        end

        # off-diagonal bounds
        offdiag_indices = [(i, j) for i in 1:ng, j in 1:ng if i != j]
        for (i, j) in offdiag_indices
            i_bus, j_bus = group[i], group[j]
            if (i_bus, j_bus) in ids(pm, nw, :buspairs)
                if bounded
                    JuMP.set_upper_bound(WR[i, j], wr_max[i_bus, j_bus])
                    JuMP.set_lower_bound(WR[i, j], wr_min[i_bus, j_bus])

                    JuMP.set_upper_bound(WI[i, j], wi_max[i_bus, j_bus])
                    JuMP.set_lower_bound(WI[i, j], wi_min[i_bus, j_bus])
                end

                # for non-semidefinite constraints
                if !((i_bus, j_bus) in visited_buspairs)
                    push!(visited_buspairs, (i_bus, j_bus))
                    var(pm, nw, :wr)[(i_bus, j_bus)] = WR[i, j]
                    var(pm, nw, :wi)[(i_bus, j_bus)] = WI[i, j]
                end
            end
        end
    end

    report && sol_component_value(pm, nw, :bus, :w, ids(pm, nw, :bus), var(pm, nw)[:w])
    report && sol_component_value_buspair(pm, nw, :buspairs, :wr, ids(pm, nw, :buspairs), var(pm, nw)[:wr])
    report && sol_component_value_buspair(pm, nw, :buspairs, :wi, ids(pm, nw, :buspairs), var(pm, nw)[:wi])
end


function constraint_model_voltage(pm::AbstractSparseSDPWRMModel, n::Int)
    _check_missing_keys(var(pm, n), [:voltage_product_groups], typeof(pm))

    pair_matrix(group) = [(i, j) for i in group, j in group]

    decomp = pm.ext[:SDconstraintDecomposition]
    groups = decomp.decomp
    voltage_product_groups = var(pm, n)[:voltage_product_groups]

    # semidefinite constraint for each group in clique grouping
    for (gidx, voltage_product_group) in enumerate(voltage_product_groups)
        _check_missing_keys(voltage_product_group, [:WR,:WI], typeof(pm))

        group = groups[gidx]
        ng = length(group)
        WR = voltage_product_group[:WR]
        WI = voltage_product_group[:WI]

        # Lower-dimensional SOC constraint equiv. to SDP for 2-vertex
        # clique
        if ng == 2
            wr_ii = WR[1, 1]
            wr_jj = WR[2, 2]
            wr_ij = WR[1, 2]
            wi_ij = WI[1, 2]
            wi_ji = WI[2, 1]

            # standard SOC form (Mosek doesn't like rotated form)
            JuMP.@constraint(pm.model, [(wr_ii + wr_jj), (wr_ii - wr_jj), 2*wr_ij, 2*wi_ij] in JuMP.SecondOrderCone())
            JuMP.@constraint(pm.model, wi_ij == -wi_ji)
        else
            JuMP.@constraint(pm.model, [WR WI; -WI WR] in JuMP.PSDCone())
        end
    end

    # linking constraints
    tree = _prim(_overlap_graph(groups))
    overlapping_pairs = [Tuple(CartesianIndices(tree)[i]) for i in (LinearIndices(tree))[findall(x->x!=0, tree)]]
    for (i, j) in overlapping_pairs
        gi, gj = groups[i], groups[j]
        var_i, var_j = voltage_product_groups[i], voltage_product_groups[j]

        Gi, Gj = pair_matrix(gi), pair_matrix(gj)
        overlap_i, overlap_j = _overlap_indices(Gi, Gj)
        indices = zip(overlap_i, overlap_j)
        for (idx_i, idx_j) in indices
            JuMP.@constraint(pm.model, var_i[:WR][idx_i] == var_j[:WR][idx_j])
            JuMP.@constraint(pm.model, var_i[:WI][idx_i] == var_j[:WI][idx_j])
        end
    end
end


"""
    adj, lookup_index = _adjacency_matrix(pm, nw)
Return:
- a sparse adjacency matrix
- `lookup_index` s.t. `lookup_index[bus_id]` returns the integer index
of the bus with `bus_id` in the adjacency matrix.
"""
function _adjacency_matrix(pm::AbstractPowerModel, nw::Int=nw_id_default)
    bus_ids = ids(pm, nw, :bus)
    buspairs = ref(pm, nw, :buspairs)

    nb = length(bus_ids)
    nl = length(buspairs)

    lookup_index = Dict((bi, i) for (i, bi) in enumerate(bus_ids))
    f = [lookup_index[bp[1]] for bp in keys(buspairs)]
    t = [lookup_index[bp[2]] for bp in keys(buspairs)]

    return sparse([f;t], [t;f], ones(2nl), nb, nb), lookup_index
end


"""
    cadj, lookup_index, ordering = _chordal_extension(pm, nw, clique_merge=false)
Return:
- a sparse adjacency matrix corresponding to a chordal extension
of the power grid graph.
- `lookup_index` s.t. `lookup_index[bus_id]` returns the integer index
of the bus with `bus_id` in the adjacency matrix.
- the graph ordering that may be used to reconstruct the chordal extension
"""
function _chordal_extension(pm::AbstractPowerModel, nw::Int, clique_merge::Bool=false)
    adj, lookup_index = _adjacency_matrix(pm, nw)
    nb = size(adj, 1)

    F = _fact_cholesky(adj, pm)
    L = sparse(F.L)
    p = F.p
    q = invperm(p)

    Rchol = L - spdiagm(0 => diag(L))
    f_idx, t_idx, V = findnz(Rchol)
    cadj = sparse([f_idx;t_idx], [t_idx;f_idx], ones(2*length(f_idx)), nb, nb)
    cadj = cadj[q, q] # revert to original bus ordering (invert cholfact permutation)
    if clique_merge
        _merge_cliques_ajd!(cadj)
        println("Clique merging completed")
    end
    return cadj, lookup_index, p
end


function _fact_cholesky(adj::SparseMatrixCSC{Float64, Int64}, ::Chordal_AMD)
    W = _adj_matrix_to_SDP(adj)
    return cholesky(W)
end

function _fact_cholesky(adj::SparseMatrixCSC{Float64, Int64}, ::Chordal_MFI)
    W = _adj_matrix_to_SDP(adj)
    perm = _minimum_fill_in_permutation(adj)
    return cholesky(W, perm=perm)
end

function _fact_cholesky(adj::SparseMatrixCSC{Float64, Int64}, ::Chordal_MD)
    W = _adj_matrix_to_SDP(adj)
    perm = _min_degree_permutation(adj)
    return cholesky(W, perm=perm)
end

function _fact_cholesky(adj::SparseMatrixCSC{Float64, Int64}, ::Chordal_MCS_M)
    W = _adj_matrix_to_SDP(adj)
    perm = _mcs(adj)
    return cholesky(W, perm=perm)
end

function _fact_cholesky(adj::SparseMatrixCSC{Float64, Int64}, ::AbstractPowerModel)
    W = _adj_matrix_to_SDP(adj)
    return cholesky(W)
end

function _adj_matrix_to_SDP(adj::SparseMatrixCSC{Float64, Int64})
    diag_el = sum(adj, dims=1)[:]
    return Hermitian(-adj + spdiagm(0 => diag_el .+ 1))
end

function _min_degree_permutation(adj::SparseMatrixCSC{Float64, Int64})
    nb = size(adj, 1)
    perm = Vector{Int}(undef, nb)
    degrees = Dict{Int, Int}()

    G = copy(adj)

    for node in 1:nb
        degrees[node] = length(findnz(G[:, node])[1])
    end

    for i in 1:nb
        # Find the node with the minimum degree
        min_degree_node = findmin(degrees)[2]

        # Add the node to the permutation
        perm[i] = min_degree_node

        # Get neighbors of the min degree node
        neighbors = Set(findnz(G[:, min_degree_node])[1])

        # Remove the node from the graph
        delete!(degrees, min_degree_node)

        # Update the adjacency matrix and degrees
        missing_edges = _missing_edges_for_clique(G, min_degree_node)
        for (ni, nj) in missing_edges
            G[ni, nj] = 1.0
            G[nj, ni] = 1.0
        end

        G[min_degree_node, :] .= 0
        G[:, min_degree_node] .= 0
        SparseArrays.dropzeros!(G)

        # Update the degrees of the neighbors
        for node in neighbors
            degrees[node] = length(findnz(adj[:, node])[1])
        end
    end
    println("MD computed")

    return perm
end

function _minimum_fill_in_permutation(adj::SparseMatrixCSC{Float64, Int64})
    nb = size(adj, 1)
    perm = Vector{Int}(undef, nb)
    Gk = copy(adj)
    
    # Initialize fill-in values
    number_of_missing_edges_per_node = Dict{Int, Int}()
    missing_edges_per_node = Dict{Int, Vector{Tuple{Int, Int}}}()
    
    for node in 1:nb
        missing_edges = _missing_edges_for_clique(Gk, node)
        missing_edges_per_node[node] = missing_edges
        number_of_missing_edges_per_node[node] = length(missing_edges)
    end
    
    for k in 1:nb
        # Find the node with the minimum fill-in
        vertex_min_missing_edges = findmin(number_of_missing_edges_per_node)[2]

        # Add the node to the permutation
        perm[k] = vertex_min_missing_edges

        # Get the missing edges for the node
        missing_edges = missing_edges_per_node[vertex_min_missing_edges]

        # Get neighbors and neighbors of neighbors
        if isempty(missing_edges)
            nodes_affected = Set(findnz(Gk[:, vertex_min_missing_edges])[1])
        else
            nodes_affected = Set(findnz(Gk[:, vertex_min_missing_edges])[1])
            for node in nodes_affected
                nodes_affected = union(nodes_affected, Set(findnz(Gk[:, node])[1]))
            end
            delete!(nodes_affected, vertex_min_missing_edges)
            for (ni, nj) in missing_edges
                Gk[ni, nj] = 1.0
                Gk[nj, ni] = 1.0
            end
        end

        # Remove the node from the graph
        delete!(number_of_missing_edges_per_node, vertex_min_missing_edges)
        delete!(missing_edges_per_node, vertex_min_missing_edges)
        Gk[vertex_min_missing_edges, :] .= 0
        Gk[:, vertex_min_missing_edges] .= 0
        SparseArrays.dropzeros!(Gk)

        # Update the degrees of the neighbors and their neighbors
        for node in nodes_affected
            missing_edges = _missing_edges_for_clique(Gk, node)
            missing_edges_per_node[node] = missing_edges
            number_of_missing_edges_per_node[node] = length(missing_edges)
        end
    end
    println("MFI computed")
    return perm
end

function _missing_edges_for_clique(adj::SparseMatrixCSC{Float64, Int64}, node::Int)
    neighbors = findnz(adj[:, node])[1]
    missing_edges = []
    for i in eachindex(neighbors)
        for j in (i+1):length(neighbors)
            ni = neighbors[i]
            nj = neighbors[j]
            if adj[ni, nj] == 0
                push!(missing_edges, (ni, nj))
            end
        end
    end
    return missing_edges
end

function _merge_cliques_ajd!(cadj::SparseMatrixCSC{Float64, Int64})
    groups = _maximal_cliques(cadj)
    weighted_clique_graph = _weighted_clique_graph(groups)
    while any(>(0), weighted_clique_graph)
        index = findfirst(>(0), weighted_clique_graph)
        i, j = Tuple(index)
        if j < i
            i, j = j, i
        end
        clique_union = union(groups[i], groups[j])
        groups[i] = clique_union
        splice!(groups, j)
        weighted_clique_graph = _merge_clique_graph(weighted_clique_graph, i, j, groups)
        for i in clique_union
            for j in clique_union
                if i != j
                    cadj[i, j] = 1.0
                end
            end
        end
    end
    return cadj
end

function _merge_clique_graph(G::SparseMatrixCSC{Int, Int}, i::Int, j::Int, groups::Vector{Vector{Int}})
    neighbors_j = findnz(G[:, j])[1]
    for node in neighbors_j
        if node != i
            G[i, node] = 1
            G[node, i] = 1
        end
    end
    merged_G = G[1:end .!= j, 1:end .!= j]
    neighbors_i = findnz(merged_G[:, i])[1]
    for node in neighbors_i
        if node != i
            weight = length(groups[i])^3 + length(groups[node])^3 - length(union(groups[i], groups[node]))^3
            merged_G[i, node] = weight
            merged_G[node, i] = weight
        end
    end
    return merged_G
end

"""
    mc = _maximal_cliques(cadj, peo)
Given a chordal graph adjacency matrix and perfect elimination
ordering, return the set of maximal cliques.
"""
function _maximal_cliques(cadj::SparseMatrixCSC, peo::Vector{Int})
    nb = size(cadj, 1)

    # use peo to obtain one clique for each vertex
    cliques = Vector(undef, nb)
    for (i, v) in enumerate(peo)
        Nv = findall(x->x!=0, cadj[:, v])
        cliques[i] = union(v, intersect(Nv, peo[i+1:end]))
    end

    # now remove cliques that are strict subsets of other cliques
    mc = Vector()
    for c1 in cliques
        # declare clique maximal if it is a subset only of itself
        if sum([issubset(c1, c2) for c2 in cliques]) == 1
            push!(mc, c1)
        end
    end
    # sort node labels within each clique
    mc = [sort(c) for c in mc]
    return mc
end
_maximal_cliques(cadj::SparseMatrixCSC) = _maximal_cliques(cadj, _mcs(cadj))

"""
    peo = _mcs(A)
Maximum cardinality search for graph adjacency matrix A.
Returns a perfect elimination ordering for chordal graphs.
"""
function _mcs(A)
    n = size(A, 1)
    w = zeros(Int, n)
    peo = zeros(Int, n)
    unnumbered = collect(1:n)

    for i = n:-1:1
        z = unnumbered[argmax(w[unnumbered])]
        filter!(x -> x != z, unnumbered)
        peo[i] = z

        Nz = findall(x->x!=0, A[:, z])
        for y in intersect(Nz, unnumbered)
            w[y] += 1
        end
    end
    return peo
end

"""
    T = _prim(A, minweight=false)
Return minimum spanning tree adjacency matrix, given adjacency matrix.
If minweight == false, return the *maximum* weight spanning tree.

Convention: start with node 1.
"""
function _prim(A, minweight=false)
    n = size(A, 1)
    candidate_edges = []
    unvisited = collect(1:n)
    next_node = 1 # convention
    T = spzeros(Int, n, n)

    while length(unvisited) > 1
        current_node = next_node
        filter!(node -> node != current_node, unvisited)

        neighbors = intersect(findall(x->x!=0, A[:, current_node]), unvisited)
        current_node_edges = [(current_node, i) for i in neighbors]
        append!(candidate_edges, current_node_edges)
        filter!(edge -> length(intersect(edge, unvisited)) == 1, candidate_edges)
        weights = [A[edge...] for edge in candidate_edges]
        next_edge = minweight ? candidate_edges[indmin(weights)] : candidate_edges[argmax(weights)]
        filter!(edge -> edge != next_edge, candidate_edges)
        T[next_edge...] = minweight ? minimum(weights) : maximum(weights)
        next_node = intersect(next_edge, unvisited)[1]
    end
    return T
end


"""
    A = _overlap_graph(groups)
Return adjacency matrix for overlap graph associated with `groups`.
I.e. if `A[i, j] = k`, then `groups[i]` and `groups[j]` share `k` elements.
"""
function _overlap_graph(groups)
    n = length(groups)
    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{Int}()
    for (i, gi) in enumerate(groups)
        for (j, gj) in enumerate(groups)
            if gi != gj
                overlap = length(intersect(gi, gj))
                if overlap > 0
                    push!(I, i)
                    push!(J, j)
                    push!(V, overlap)
                end
            end
        end
    end
    return sparse(I, J, V, n, n)
end


function _filter_flipped_pairs!(pairs)
    for (i, j) in pairs
        if i != j && (j, i) in pairs
            filter!(x -> x != (j, i), pairs)
        end
    end
end


"""
    idx_a, idx_b = _overlap_indices(A, B)
Given two arrays (sizes need not match) that share some values, return:

- linear index of shared values in A
- linear index of shared values in B

Thus, A[idx_a] == B[idx_b].
"""
function _overlap_indices(A::Array, B::Array, symmetric=true)
    overlap = intersect(A, B)
    symmetric && _filter_flipped_pairs!(overlap)
    idx_a = [something(findfirst(isequal(o), A), 0) for o in overlap]
    idx_b = [something(findfirst(isequal(o), B), 0) for o in overlap]
    return idx_a, idx_b
end


"""
    ps = _problem_size(groups)
Returns the sum of variables and linking constraints corresponding to the
semidefinite constraint decomposition given by `groups`. This function is
not necessary for the operation of clique merge, since `merge_cost`
computes the change in problem size for a proposed group merge.
"""
function _problem_size(groups)
    nvars(n::Integer) = n*(2*n + 1)
    A = _prim(_overlap_graph(groups))
    return sum(nvars.(Int.(nonzeros(A)))) + sum(nvars.(length.(groups)))
end

function _weighted_clique_graph(groups)
    n = length(groups)
    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{Int}()
    for (i, gi) in enumerate(groups)
        for (j, gj) in enumerate(groups)
            if gi != gj
                if length(intersect(gi, gj)) > 0
                    weight = length(gi)^3 + length(gj)^3 - length(union(gi, gj))^3
                    push!(I, i)
                    push!(J, j)
                    push!(V, weight)
                end
            end
        end
    end
    return sparse(I, J, V, n, n)
end
