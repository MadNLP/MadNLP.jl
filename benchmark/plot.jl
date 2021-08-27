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
    mem = Dict()
    iter = Dict()

    name = readdlm("name-$class.csv")[:]
    for solver in SOLVERS
        status[solver] = readdlm("status-$class-$solver.csv")[:]
        time[solver] = readdlm("time-$class-$solver.csv")[:]
        mem[solver] = readdlm("mem-$class-$solver.csv")[:]
        iter[solver] = readdlm("iter-$class-$solver.csv")[:]
    end

    for (str,metric) in [("time",time),("iterations",iter),("memory",mem)]
        relmetric = deepcopy(metric)
        
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
                fastest = min(fastest,metric[solver][i])
            end

            for solver in SOLVERS
                if status[solver][i] == 3
                    relmetric[solver][i] = NaN
                else
                    relmetric[solver][i] = log2(relmetric[solver][i]/fastest)
                end
            end
        end

        for solver in SOLVERS
            filter!(a->!isnan(a),relmetric[solver])
        end

        p = plot(;
                 ylim=(0,1),
                 xlim=(
                     minimum(minimum(relmetric[solver]) for solver in SOLVERS),
                     maximum(maximum(relmetric[solver]) for solver in SOLVERS)
                 ),
                 xlabel="Not More Than 2ˣ-Times Worse Than Best Solver ($str)",
                 ylabel="Fraction of Problems Solved",
                 framestyle=:box,
                 legend=:bottomright)
        
        for solver in SOLVERS
            y = [0:length(relmetric[solver]);length(relmetric[solver])]/length(name)
            push!(relmetric[solver],maximum(maximum(relmetric[s]) for s in SOLVERS))
            push!(relmetric[solver],0)
            plot!(p,sort(relmetric[solver]),y;
                  qqline=:none,
                  linetype=:steppost,
                  label=LABELS[solver])
        end
        savefig(p,"$str-$class.pdf")
    end
end
