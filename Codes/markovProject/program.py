import markovDelay as mD
data = []
initialGuess = [60,0.3,0.3,1,1,1]
initialDeviations = [0.5,0.1,0.1,0.3,0.3,0.3]

data.append([0.68, 0.77,0.31,0.42,0.89])
data.append([0.96,2.32,1.72,1.72,1.59,2.51])
data.append([ii*3600*24 for ii in [10.5, 255.675, 11.7,25.0, 47, 26]])
data.append([0.24, 2.095, 1.147, 1.397, 1.14, 0.67])
data.append([0.10, 1.099,0.950,0.950,0.25,0.32])

dataTmp = []
for ii in range(5):
    tmpL = []
    for jj in range(5):
        tmpL.append(data[jj][ii])
    dataTmp.append(tmpL)

data = dataTmp

result = mD.metropolis_Hastings(initialGuess, 1000, initialDeviations, data)
print(result[0])
print(result[1])
