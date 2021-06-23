@test begin
    local m,x
    m=Model(MadNLP.Optimizer)
    @variable(m,x)
    @objective(m,Min,x^2)
    MOIU.attach_optimizer(m)

    nlp = MadNLP.NonlinearProgram(m.moi_backend.optimizer.model)
    ips = MadNLP.Solver(nlp)
    
    show(stdout, "text/plain",nlp)
    show(stdout, "text/plain",ips)
    true
end
