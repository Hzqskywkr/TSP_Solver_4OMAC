import numpy as np
import matplotlib.pyplot as plt
from omacMatmul.matmul import oMAC_matmul
import time
from utils import read_TSP
from utils import read_TSP_graph
from utils import Q_Matrix_1,Q_Matrix_2,Q_Matrix_3, quad_K
import random

TSP_files = ['test10.xml','burma14.xml','ulysses16.xml']
TSP_graph = ['test10.tsp','burma14.tsp','ulysses16.tsp']
num = 0
n,cost1,cost2 = read_TSP(TSP_files[num])
startTime_model = time.perf_counter()
n_list = range(1,n+1)
cost = []
for i in range(n*(n-1)):
    temp = cost1[i]*10**(cost2[i])
    cost.append(temp)
weight = np.zeros((n,n))
for i in range(n):
    temp = cost[(n-1)*i:(n-1)*i+n-1]
    temp.insert(i,0)
    weight[i,:]=np.array(temp)
A = np.max(weight)*1
QA = Q_Matrix_1(n)
edges = []
for i in range(n):
    for j in range(i+1,n):
        edges.append([i,j])
QA = A*QA
QB = np.zeros((n * n, n * n))
QB = Q_Matrix_3(n,QB,edges,weight)
Q = QA+QB
Qmax = abs(Q).max()
Q = Q/Qmax
Const = A*2*n/Qmax

FIXED_SEED = 0
MATRIX_SIZE = n * n
endTime_model = time.perf_counter()
t_model = endTime_model - startTime_model
print('time for model',t_model)

def getThresholds(K, L):
    matrix = np.zeros((K.shape[0]))
    for i in range(K.shape[0]):
        matrix[i] = (K[i].sum() - L[i])/2
    return matrix

def SvectorInitialization(S):
    for i in range(S.shape[0]):
        val = np.random.randint(0, 2 ** 15) % 2
        S[i] = val

    return S

def isVectorZero(S):
    return np.all(S == 0)

def isVectorOne(S):
    return np.all(S == 1)

def getKL(Q):
    # This function get the matrix K and external field L
    # Calculate K matrix for Q Matrix
    K = Q.copy()
    K = -0.5 * K
    for i in range(MATRIX_SIZE):
        K[i][i] = 0
    L = np.zeros((MATRIX_SIZE, 1))  # external magnetic field
    L = np.sum(Q, axis=1)
    L = -0.5 * L

    return K, L

def findminindex(DH):
    minDH = np.min(DH)
    index_minSigma = []
    for i in range(DH.shape[0]):
        if DH[i] == minDH:
            index_minSigma.append(i)
    return index_minSigma

def Flip(S,S_PIC, thresholds):
    Sigma = 2 * S - 1
    DM = S_PIC.T - thresholds
    DH = Sigma.T * DM #(-2)*Sigma'*DM
    minSigma = findminindex(DH)
    index = np.random.choice(minSigma)
    S[index] = 1-S[index]
    return S

def check(S):
    rout = np.zeros(n, dtype=np.int)
    hard = 0
    ind_check = []
    for i in range(0, MATRIX_SIZE, n):
        x = S.T[i:i + n]
        none = np.sum(x == 1)
        if none == 1:
            ind = np.where(x == 1)[0][0]
            if ind in ind_check:
                hard = 1
                return hard,rout
            else:
                ind_check.append(ind)
                rout[ind] = int(i / n)
        else:
            hard = 1
            return hard,rout
    return hard,rout

def Distance(rout, weight):
    distance = 0
    for i in range(n):
        if i == n-1:
            distance += weight[rout[i], rout[0]]
        else:
            distance += weight[rout[i],rout[i+1]]
    return distance

def runIsingAlgorithm_simulator(Q):
    startTime_ini = time.perf_counter()
    K, L = getKL(Q)
    scale, KS = quad_K(K)
    S = np.zeros((256), dtype=np.int)
    SvectorInitialization(S)
    # Calculate initial energy
    best_matrix = S
    best_energy = 1E9
    # Using the adjacency matrix to set the thresholds
    thresholds = getThresholds(K, L)
    KS_Pace = np.pad(KS,((0,256-MATRIX_SIZE),(0,256-MATRIX_SIZE)),'constant',constant_values = 0)
    KS_Pace = KS_Pace.T.reshape(1,256,256)
    scale_S = 8
    niter = 40000
    omatmul = oMAC_matmul()
    omatmul.init()
    endTime_ini = time.perf_counter()
    t_ini = endTime_ini - startTime_ini
    t_MVM = 0
    t_energy = 0
    t_record = 0
    t_update = 0
    startTime = time.perf_counter()
    # start the iteration
    for i in range(niter):
        S_Pace = scale_S*S.reshape(1,1,256)
        startTime_MVM = time.perf_counter()
        S_PIC, latency = omatmul(S_Pace, KS_Pace)
        endTime_MVM = time.perf_counter()
        t_MVM += endTime_MVM - startTime_MVM
        S_PIC = S_PIC.reshape(256,)
        S_PIC = S_PIC/(scale*scale_S)
        # check abd update the best energyi
        startTime_eng = time.perf_counter()
        
        hard,rout = check(S[0:MATRIX_SIZE])
        if not hard:
            energy = Distance(rout,weight)
            startTime_record = time.perf_counter()
            if energy < best_energy and not isVectorZero(S) and not isVectorOne(S):
                best_matrix = S.copy()
                best_energy = energy
                #print("New best")
                #print(f"Iteration {i}")
                #print(best_matrix.T)
                #print(best_energy)
            endTime_record = time.perf_counter()
            t_record +=  endTime_record - startTime_record
        endTime_eng = time.perf_counter()
        t_energy += endTime_eng - startTime_eng
        # Updata the state S
        startTime_update = time.perf_counter()
        S[0:MATRIX_SIZE] = Flip(S[0:MATRIX_SIZE], S_PIC[0:MATRIX_SIZE], thresholds)
        S = S.astype(int)
        endTime_update = time.perf_counter()
        t_update += endTime_update - startTime_update

    endTime = time.perf_counter()
    d = endTime - startTime
    t_periter_omac = t_MVM/i
    t_periter_energy = t_energy/i
    t_periter_record = t_record/i
    t_periter_update = t_update/i
    print(f"caling {niter} iterations: {i} s")
    print('time for initial',t_ini)
    print('time for MVM', t_MVM)
    print('time for energy', t_energy)
    print('time for record', t_record)
    print('time for update', t_update)
    print("time only for loops:")
    print(f"average time per iterations: {d  / i} s")
    print("average time per iterations omac", t_periter_omac)
    print("average time per iterations energy", t_periter_energy)
    print("average time per iterations record", t_periter_record)
    print("average time per iterations update", t_periter_update)
    return best_matrix, best_energy

def main():
    seed = int(time.time())
    np.random.seed(seed)
    print("start main")
    startTime = time.perf_counter()
    best_matrix, best_energy = runIsingAlgorithm_simulator(Q)
    endTime = time.perf_counter()
    hard,rout = check(best_matrix[0:MATRIX_SIZE])
    if hard == 1:
        print('hard constraints not sat ')
    else:
        print('rout is',rout)
        distance = Distance(rout,weight)
        print('distance',distance)
    Xrout = []
    Yrout = []
    X, Y = read_TSP_graph(TSP_graph[num])
    for pos in rout:
        Xrout.append(X[pos])
        Yrout.append(Y[pos])
    Xrout.append(X[rout[0]])
    Yrout.append(Y[rout[0]])
    plt.scatter(X, Y)
    plt.plot(Xrout,Yrout,'--')
    plt.show()
    print("Done")

if __name__ == "__main__":
    main()
