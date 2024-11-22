# 11/11/24      Michael Nunez  Robustness test when true relationship is sigmoidal
# 12/11/24      Michael Nunez  Robustness test 2 without true relationship
alpha = runif(1000, 0.8, 1.4)
extdata = alpha + rnorm(length(alpha),0, 0.1)
extdata2 = 0.6/(1+exp(-32*(alpha-1.1))) + 0.8 + rnorm(length(alpha),0,0.1)

var(extdata) # Should be 0.04
var(extdata2)

unif_a = 0.75
unif_b = unif_a + sqrt(0.48)
unif_var = (1/12)*(unif_b-unif_a)^2
unif_sd = sqrt(unif_var)
extdata3 = runif(length(alpha), unif_a, unif_b)

x11()
plot(alpha, extdata)
lines(alpha, extdata2,type='p',col='red')
lines(alpha, extdata3,type='p',col='blue')