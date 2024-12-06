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

function write_rdf(outname::AbstractString, rdf::Vector{Float64}, system_params::SystemParameters)
    bins = [bin * system_params.bin_width for bin in 1:(system_params.n_bins)]

    open(outname, "w") do io
        println(io, "# System: $(system_params.system_name)")
        println(io, "# RDF data ($(system_params.atom_name) - $(system_params.atom_name))")
        println(io, "# r, Å; g(r);")
        for (bin, g) in zip(bins, rdf)
            println(io, @sprintf("%6.4f %12.4f", bin, g))
        end
    end

    check_file(outname)
end

function write_energies(outname::AbstractString, energies::Vector{Float64},
                        mc_params::MonteCarloParameters, system_params::SystemParameters,
                        slicing::Int=1)
    steps = 0:(mc_params.output_frequency * slicing):(mc_params.steps)
    sliced_energies = energies[1:slicing:end]

    open(outname, "w") do io
        println(io, "# System: $(system_params.system_name)")
        println(io, @sprintf("# %8s %22s", "Step", "Total energy, kJ/mol"))
        for (step, energy) in zip(steps, sliced_energies)
            println(io, @sprintf("%9d %10.4f", step, energy))
        end
    end

    check_file(outname)
end

function write_trajectory(conf::Chemfiles.ChemfilesArray, box::Vector{Float64},
                          system_params::SystemParameters, outname::AbstractString,
                          mode::Char='w')
    frame = Frame()
    box_center = box ./ 2
    set_cell!(frame, UnitCell(box))

    for i in 1:(system_params.n_atoms)
        wrapped_atom_coords = wrap!(UnitCell(frame), conf[:, i]) .+ box_center
        add_atom!(frame, Atom(system_params.atom_name), wrapped_atom_coords)
    end

    Trajectory(outname, mode) do traj
        write(traj, frame)
    end

    check_file(outname)
end

function read_rdf(rdfname::AbstractString)
    check_file(rdfname)

    bins = Float64[]
    rdf = Float64[]

    open(rdfname, "r") do file
        for line in eachline(file)
            stripped_line = strip(line)
            isempty(stripped_line) && continue  # Skip empty lines
            startswith(stripped_line, "#") && continue  # Skip comment lines

            values = split(stripped_line)
            length(values) < 2 && continue  # Skip lines with insufficient data

            push!(bins, parse(Float64, values[1]))
            push!(rdf, parse(Float64, values[2]))
        end
    end

    isempty(bins) && @warn "No data found in $rdfname"

    return bins, rdf
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
    min_step_size = 0.5  # å
    min_box_length = minimum(box)
    max_step_size = min_box_length / 2.0

    # Calculate current acceptance ratio
    acceptance_ratio = accepted_moves / mc_params.step_adjust_frequency

    # Adjust step size based on target acceptance ratio
    new_step_size = (acceptance_ratio / system_params.target_acceptance_ratio) * current_step_size

    # Enforce step size bounds
    return clamp(new_step_size, min_step_size, max_step_size)
end
