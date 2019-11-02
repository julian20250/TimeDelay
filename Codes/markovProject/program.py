import markovDelay as mD
from numpy import pi

data = []
initialGuess = [60,0.3,0.3,1,1,1]
initialDeviations = [0.5,0.01,0.01,0.5,0.5,0.5]
errorDt = [ii*3600*24 for ii in [0.2, 36.525, 1.2, 1.6, 6,5]]
data.append([0.68, 0.77,0.31,0.42,0.89])
data.append([0.96,2.32,1.72,1.72,1.59,2.51])
data.append([ii*3600*24 for ii in [10.5, 255.675, 11.7,25.0, 47, 26]]) #To seconds
data.append([ii/3600*pi/180 for ii in [0.24, 2.095, 1.147, 1.397, 1.14, 0.67]]) #To radians
data.append([ii/3600*pi/180 for ii in [0.10, 1.099,0.950,0.950,0.25,0.32]]) #To radians

dataTmp = []
for ii in range(5):
    tmpL = []
    for jj in range(5):
        tmpL.append(data[jj][ii])
    dataTmp.append(tmpL)

data = dataTmp

result = mD.metropolis_Hastings(initialGuess, 1000, initialDeviations, data,
        errorDt, 10)
mD.graph_Likelihood(result[2])
chain=input("Wish to burn? (y/n)>")
while chain!="n":
    until = int(input("Until which value > "))
    tmpResult= result.copy()
    result[0],result[2]= mD.burn_Result(tmpResult[0], until), mD.burn_Result(tmpResult[2], until)
    mD.graph_Likelihood(result[2])
    chain = input("Continue burning? (y/n)>")

mD.graph_Confidence(result[0], data, errorDt)
print(result[0][-1])
