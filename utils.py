import numpy as np
import re
import string

def read_TSP(TSP_file):
    with open(TSP_file) as f:
        print(f"[READ] {TSP_file}")
        n = 0
        cost1 = []
        cost2 = []
        for line in f.readlines():
            line = line.strip(' \n')
            #print('line',line)
            g = re.search("\<edge *", line)
            if len(line)==0:
                continue
            if line == '<vertex>':
                n += 1
            if g:
                cost = line.split("\"")[1]
                cost_split = cost.split('e+')
                cost1.append(float(cost_split[0]))
                cost2.append(int(cost_split[1]))
                #print('cost',cost)

    return n,cost1, cost2

def read_TSP_graph(TSP_graph):
    x = []
    y = []
    with open(TSP_graph) as f:
        print(f"[READ] {TSP_graph}")
        for line in f.readlines():
            line = line.strip(' \n')
            if line.startswith('D'):
                num_list = line.split(':')
                if num_list[0] == 'DIMENSION':
                    n = int(num_list[1])
            if line[0].isdigit():

                num_list = [x for x in line.split(' ') if x]
                x.append(float(num_list[1]))
                y.append(float(num_list[2]))

    return x, y

def Q_Matrix_1(n):
    Q = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            Q[i * n + j, i * n + j] += -1
            for k in range(j + 1, n):
                Q[i * n + j, i * n + k] += 2

    for i in range(n):
        for j in range(n):
            Q[j*n+i,j*n+i] += -1
            for k in range(j+1,n):
                Q[j * n + i, k * n + i] += 2

    Q = (Q+Q.T)/2

    return Q

def Q_Matrix_2(n, Q, edge_null):
    for edge in edge_null:
        a = edge[0]
        b = edge[1]
        for k in range(n):
            if k == n-1:
                Q[a * n + k, b * n] += 1
                Q[b * n + k, a * n] += 1
            else:
                Q[a*n+k,b*n+k+1] += 1
                Q[b * n + k, a * n + k + 1] += 1
    Q = (Q + Q.T) / 2

    return Q

def Q_Matrix_3(n, Q, edges, weight):
    for edge in edges:
        a = edge[0]
        b = edge[1]
        for k in range(n):
            if k == n-1:
                Q[a * n + k, b * n] += 1*weight[a,b]
                Q[b * n + k, a * n] += 1 * weight[b, a]
            else:
                Q[a*n+k,b*n+k+1] += 1*weight[a,b]
                Q[b * n + k, a * n + k + 1] += 1 * weight[b, a]
    Q = (Q + Q.T) / 2

    return Q

def quad_K(K):
    Bit = 4
    SL = -2**(Bit-1)
    SR = 2**(Bit-1)-1
    KL = K.min()
    KR = K.max()
    epsilon = 1E-6
    if KR<0:
        scale = SL/KL
    elif KL>0:
        scale = SR/KR
    else:
        scale = min(SL/(KL+epsilon),SR/(KR+epsilon))
    KS = np.around(K*scale)

    return scale, KS
