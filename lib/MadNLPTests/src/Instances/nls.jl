F(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]

function NLSModel()
    x0 = [-1.2; 1.0]
    return ADNLSModel(F, x0, 2)
end
