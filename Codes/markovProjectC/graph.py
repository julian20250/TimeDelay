import graphicLibraries as gL

errorDt = [ii*3600*24 for ii in [0.2, 36.525, 1.2, 1.6, 6,5]]
likelihood, accepted, rejected = gL.getValues()
gL.graph_Likelihood(likelihood)

chain=input("Wish to burn? (y/n)>")
while chain!="n":
    until = int(input("Until which value > "))
    likelihood= gL.burn_Result(likelihood, until)
    accepted = gL.burn_Result(accepted, until)
    gL.graph_Likelihood(likelihood)
    chain = input("Continue burning? (y/n)>")

gL.graph_Confidence(accepted, errorDt)
