@with_kw mutable struct Logger
    print_level::LogLevels = INFO
    file_print_level::LogLevels = INFO
    file::Union{IOStream,Nothing} = nothing
end

get_parent(logger::Logger) = logger.parent
get_level(logger::Logger) = logger.print_level
get_file_level(logger::Logger) = logger.file_print_level
get_file(logger::Logger) = logger.file
finalize(logger::Logger) = logger.file != nothing && close(f)

for (name,level,color) in [(:trace,TRACE,7),(:debug,DEBUG,6),(:info,INFO,256),(:notice,NOTICE,256),(:warn,WARN,5),(:error,ERROR,9)]
    @eval begin
        macro $name(logger,str)
            gl = $get_level
            gfl= $get_file_level
            gf = $get_file
            l = $level
            c = $color
            code = quote
                if $gl($logger) <= $l
                    if $c == 256
                        println($str)
                    else
                        printstyled($str,"\n",color=$c)
                    end
                end
                if $gf($logger) != nothing && $gfl($logger) <= $l
                    println($gf(logger),$str)
                end
            end
            esc(code)
        end
    end
end
