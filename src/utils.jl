using ..ML_IMC

function check_file(filename::AbstractString)
    isfile(filename) || throw(ArgumentError("Could not locate file: $filename"))
end

function read_xtc(system_params::SystemParameters)
    check_file(system_params.trajectory_file)
    return Trajectory(system_params.trajectory_file)
end

function read_pdb(system_params::SystemParameters)
    check_file(system_params.topology_file)
    return Trajectory(system_params.topology_file)
end

function write_rdf(outname::AbstractString, rdf::AbstractVector{T},
                   system_params::SystemParameters) where {T <: AbstractFloat}
    bin_width = T(system_params.bin_width)
    bin_centers = [(T(bin) - T(0.5)) * bin_width for bin in 1:(system_params.n_bins)]

    open(outname, "w") do io
        println(io, "# System: $(system_params.system_name)")
        println(io, "# RDF data ($(system_params.atom_name) - $(system_params.atom_name))")
        println(io, "# r_center, Å; g(r);")
        for (bin_center, g) in zip(bin_centers, rdf)
            println(io, @sprintf("%.17e %.17e", bin_center, g))
        end
    end

    check_file(outname)
end

function write_energies(outname::AbstractString, energies::AbstractVector{<:AbstractFloat},
                        mc_params::MonteCarloParameters, system_params::SystemParameters,
                        slicing::Int=1)
    steps = 0:(mc_params.output_frequency * slicing):(mc_params.steps)
    sliced_energies = energies[1:slicing:end]

    open(outname, "w") do io
        println(io, "# System: $(system_params.system_name)")
        println(io, @sprintf("# %8s %22s", "Step", "Total energy, kJ/mol"))
        for (step, energy) in zip(steps, sliced_energies)
            println(io, @sprintf("%9d %.17e", step, energy))
        end
    end

    check_file(outname)
end

function write_trajectory(conf::AbstractMatrix{T}, box::AbstractVector{T},
                          system_params::SystemParameters, outname::AbstractString,
                          mode::Char='w') where {T <: AbstractFloat}
    frame = Frame()
    box_center = box ./ T(2)
    # Chemfiles' write API accepts Cdouble arrays; simulation state remains Float32.
    cell = UnitCell(Cdouble.(box))
    set_cell!(frame, cell)

    for i in 1:(system_params.n_atoms)
        wrapped_atom_coords = wrap!(cell, Cdouble.(conf[:, i])) .+ Cdouble.(box_center)
        add_atom!(frame, Atom(system_params.atom_name), wrapped_atom_coords)
    end

    Trajectory(outname, mode) do traj
        write(traj, frame)
    end

    check_file(outname)
end

function read_rdf(rdfname::AbstractString)
    check_file(rdfname)

    bins = Float32[]
    rdf = Float32[]

    open(rdfname, "r") do file
        for line in eachline(file)
            stripped_line = strip(line)
            isempty(stripped_line) && continue  # Skip empty lines
            startswith(stripped_line, "#") && continue  # Skip comment lines

            values = split(stripped_line)
            length(values) < 2 && continue  # Skip lines with insufficient data

            push!(bins, parse(Float32, values[1]))
            push!(rdf, parse(Float32, values[2]))
        end
    end

    isempty(bins) && @warn "No data found in $rdfname"

    return bins, rdf
end

function infer_bin_width(bins::AbstractVector{T}; atol::T=convert(T, 1.0f-8))::T where {T <: AbstractFloat}
    isempty(bins) && throw(ArgumentError("RDF bins vector cannot be empty"))

    if length(bins) == 1
        bins[1] > zero(T) || throw(ArgumentError("RDF bin width must be positive"))
        return bins[1]
    end

    bin_width = bins[2] - bins[1]
    bin_width > zero(T) || throw(ArgumentError("RDF bins must be strictly increasing"))

    for i in 3:length(bins)
        step = bins[i] - bins[i - 1]
        if !isapprox(step, bin_width; atol=atol, rtol=sqrt(eps(T)))
            throw(ArgumentError("RDF bins must be uniformly spaced"))
        end
    end

    return bin_width
end

function compute_adaptive_gradient_coefficients(system_losses::AbstractVector{T})::Vector{T} where {T <: AbstractFloat}
    isempty(system_losses) && throw(ArgumentError("System losses vector cannot be empty"))

    max_loss = maximum(system_losses)
    if iszero(max_loss)
        n_systems = length(system_losses)
        return fill(one(T) / n_systems, n_systems)
    end

    # Normalize losses relative to maximum loss
    relative_coefficients = system_losses ./ max_loss

    # Scale coefficients to sum to 1.0
    normalization_factor = one(T) / sum(relative_coefficients)

    return relative_coefficients .* normalization_factor
end

function adjust_monte_carlo_step!(current_step_size::T,
                                  system_params::SystemParameters,
                                  box::AbstractVector{T},
                                  mc_params::MonteCarloParameters,
                                  accepted_moves::Integer)::T where {T <: AbstractFloat}
    # Calculate minimum and maximum step sizes based on box dimensions
    min_step_size = T(0.1)  # Å - lowered for better flexibility
    min_box_length = minimum(box)
    max_step_size = min_box_length * T(0.3) # Tightened from 0.5 to 0.3

    # Calculate current acceptance ratio
    acceptance_ratio = T(accepted_moves) / T(mc_params.step_adjust_frequency)
    target = T(system_params.target_acceptance_ratio)

    # Use a damped adjustment to handle noisy models
    # Formula: new = old * (1 + damping * (acc - target) / target)
    damping = T(0.2)
    scaling = one(T) + damping * (acceptance_ratio - target) / target
    new_step_size = current_step_size * scaling

    # Enforce step size bounds
    return clamp(new_step_size, min_step_size, max_step_size)
end
