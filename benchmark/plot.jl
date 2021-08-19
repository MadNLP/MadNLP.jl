using Plots, DelimitedFiles

const CLASSES = filter(e-> e in ["cutest","power"], ARGS)
const SOLVERS = filter(e-> e in ["current","master","ipopt","knitro"], ARGS)

const LABELS = Dict(
    "current" => "MadNLP (dev)",
    "master" => "MadNLP (master)",
    "ipopt" => "Ipopt",
    "knitro" => "Knitro",
)

for class in CLASSES
    
    time = Dict()
    status = Dict()

    name = readdlm("name-$class.csv")[:]
    for solver in SOLVERS
        time[solver] = readdlm("time-$class-$solver.csv")[:]
        status[solver] = readdlm("status-$class-$solver.csv")[:]
    end
    
    reltime = deepcopy(time)

    for i=1:length(name)
        top = []
        topstatus = 3
        for solver in SOLVERS
            if status[solver][i] < topstatus
                empty!(top)
                push!(top,solver)
                topstatus = status[solver][i]
            elseif status[solver][i] == topstatus
                push!(top,solver)
            end
        end

        fastest = Inf
        for solver in top
            fastest = min(fastest,time[solver][i])
        end

        for solver in SOLVERS
            if status[solver][i] == 3
                reltime[solver][i] = NaN
            else
                reltime[solver][i] = log2(reltime[solver][i]/fastest)
            end
        end
    end

    for solver in SOLVERS
        filter!(a->!isnan(a),reltime[solver])
    end

    p = plot(;
             ylim=(0,1),
             xlim=(
                 minimum(minimum(reltime[solver]) for solver in SOLVERS),
                 maximum(maximum(reltime[solver]) for solver in SOLVERS)
             ),
             xlabel="Not More Than 2ˣ-Times Worse Than Best Solver",
             ylabel="Fraction of Problems Solved",
             framestyle=:box,
             legend=:bottomright)
    for solver in SOLVERS
        y = [1:length(reltime[solver]);length(reltime[solver])]/length(name)
        push!(reltime[solver],maximum(maximum(reltime[s]) for s in SOLVERS))
        plot!(p,sort(reltime[solver]),y;
              qqline=:none,
              linetype=:steppost,
              label=LABELS[solver])
    end
    savefig(p,"$class.pdf")
end
