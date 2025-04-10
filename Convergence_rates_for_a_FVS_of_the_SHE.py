#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[ ]:


import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt 


#This programm is used to calculate convergence rates for a finite volume scheme of the stochastic heat equation on the domain (-1,1)^2 and
#time interval [0,T].
#In space we subdivide (-1,1)^2 into subsquares, each having length 2/L_max or 2/L for L in L_list, respectively.
#The time interval [0,T] is decomposed in equidistant intervals of length T/N_max or T/n_iter for n_iter in N_list, respectively.
#For a detailed description of the problem tackled please see https://arxiv.org/abs/2404.05655

T = 1 #End time
L_max = 32 # Number of squares in each spacial direction for U_max
L_list = [4, 8] #Number of squares in each spacial direction for U_appr
N_p = 1000  # Necessarily a multiply of Q below!!!!
N_max = 4096 #Number of time steps for U_max. Necessarily a multiply of all members in N_list!!!
N_list = [64, 128, 256]  #Number of time steps for U_appr
Q = 10 #Number of paths for which the solution should be calculated parallely. 
#The higher Q, the faster, but more memory consumption. 

def Brownian_motion(N_p, N_max, T): #calculates the increments of the paths of the BM with step size T/N_max
    rng = np.random.default_rng()
    Z = np.sqrt(T / N_max) * rng.normal(loc=0.0, scale=1.0, size=(N_p, N_max))
    
    return Z

Z = Brownian_motion(N_p, N_max, T)

def Brownian_motion_iter(N_p, N_list, T): #calculates the increments of the paths of the BM with larger step size T/N_list
    Y = {}
    for n_iter in N_list:
        l = N_max // n_iter
        Y[n_iter] = np.zeros((N_p, n_iter))
        for j in range(n_iter):
            for k in range(N_p):
                for i in range(l):
                    Y[n_iter][k][j] += Z[k, j*l + i]
    
    return Y
Y = Brownian_motion_iter(N_p, N_list, T)

print("Calculation Brownian motions done")

def mesh_and_FV_matrix(L, N, T):

    def mesh(L): #identifying the neighbours of each cell. The cells are numbered in the following:
                #bottom left cell is the 0th cell, then going upwards. The top left cell is the (L-1)th cell.
                #Then going one step to the right and starting with the bottom cell and going upwards.
                #i.e., the (i*L+j)th cell is the cell i steps to the right and j steps to the top.
        cells_neighbours = []
        
        for i in range(L):
            for j in range(L):
                neighbours = []
                if i != 0:  # left neighbour
                    neighbours.append((i-1) * L + j)
                if j != 0:  # lower neighbour
                    neighbours.append(i * L + j - 1)
                if j != L - 1:  # upper neighbour
                    neighbours.append(i * L + j + 1)
                if i != L - 1:  # right neighbour
                    neighbours.append((i+1) *  L + j)
                cells_neighbours.append(neighbours)
                
        return cells_neighbours
    
    B = mesh(L) 
    
    def FV_matrix(N, T, L): #calculating the finite volume matrix
        A = np.zeros((L**2, L**2))
        
        for i in range(L):
            for j in range(L):
                A[i * L + j][i * L + j] = 4 / L**2 + len(B[i * L + j]) * T / N
                
                if i != 0:  # west neighbour
                    A[i * L + j][(i-1) * L + j] = -T / N
                if j != 0:  # south neighbour
                    A[i * L + j][i * L + j - 1] = -T / N
                if j != L - 1:  # north neighbour
                    A[i * L + j][i * L + j + 1] = -T / N
                if i != L - 1:  # east neighbour
                    A[i * L + j][(i+1) * L + j] = -T / N
        
        return csr_matrix(A)  # Return the sparse representation directly
    
    return FV_matrix(N, T, L)

FVM_L_max_N_max = mesh_and_FV_matrix(L_max, N_max, T)  #finite volume matrix for L_max and N_max
identity_matrix_L_max = np.eye(L_max**2)  #identity matrix of size L_max**2
inverse_matrix_L_max_N_max = spsolve(FVM_L_max_N_max, identity_matrix_L_max) #inverse of the finite volume matrix for L_max and N_max

identity_matrix_L_list = {} #collection of identity matrices of the size according to L in L_list

for L in L_list:

    identity_matrix_L_list[L] = np.eye(L**2)  



def inverse_matrix_iterations(L, N_list, T):
    matrix = {}
    for n_iter in N_list:

        matrix[n_iter] = spsolve(mesh_and_FV_matrix(L, n_iter, T), identity_matrix_L_list[L])

    return matrix

inverse_matrix_iter = {} #collection of the inverses of the finite volume matrices for L in L_list and n_iter in N_list

for L in L_list:
    inverse_matrix_iter[L] = inverse_matrix_iterations(L, N_list, T)

print("Calculation inverse matrices done")

def initial_value_not_equal_zero(L, Q, N_max): #Calculating the FV-approximation of the initial value u_0(x,y) = C_1(x) * C_2(y),
                                                #where C_1'(x) = 3(x^2-1), C_2'(y)= 3(y^2-1)
    u_h = np.zeros((L**2, Q, N_max + 1))

    x = np.linspace(-1, 1, L, endpoint=False)  
    y = np.linspace(-1, 1, L, endpoint=False)

    A_x, A_y = np.meshgrid(x, y)
    B_x, B_y = np.meshgrid(x + 2 / L, y + 2 / L)

    value = (L**2 / 4) * (
        (1 / 16) * (B_y.flatten()**4 - A_y.flatten()**4) * (B_x.flatten()**4 - A_x.flatten()**4) -
        (3 / 8) * (B_y.flatten()**4 - A_y.flatten()**4) * (B_x.flatten()**2 - A_x.flatten()**2) -
        (3 / 8) * (B_y.flatten()**2 - A_y.flatten()**2) * (B_x.flatten()**4 - A_x.flatten()**4) +
        (9 / 4) * (B_y.flatten()**2 - A_y.flatten()**2) * (B_x.flatten()**2 - A_x.flatten()**2)
    )
    u_h[:, :, 0] = value[:, np.newaxis]  

    return u_h

def initial_value_not_equal_zero_no_symmetry(L, Q, N_max):#Calculating the FV-approximation of the initial value u_0(x,y) = C_1(x) * C_2(y),
                                                #where C_1'(x) = (x^2-1)(x+3), C_2'(y) = (y^2-1)(y-2)
    u_h = np.zeros((L**2, Q, N_max + 1))

    x = np.linspace(-1, 1, L, endpoint=False)  
    y = np.linspace(-1, 1, L, endpoint=False)

    A_x, A_y = np.meshgrid(x, y)
    B_x, B_y = np.meshgrid(x + 2 / L, y + 2 / L)

    value = (L**2 / 4) * (1/16)*(((1/5) * (B_x.flatten()**5 - A_x.flatten()**5) + (B_x.flatten()**4 - A_x.flatten()**4) - 
                          (2/3)*(B_x.flatten()**3 - A_x.flatten()**3) - 6*(B_x.flatten()**2 - A_x.flatten()**2)) *
                          ((1/5) * (B_y.flatten()**5 - A_y.flatten()**5) - (2/3)*(B_y.flatten()**4 - A_y.flatten()**4) - 
                          (2/3)*(B_y.flatten()**3 - A_y.flatten()**3) + 4*(B_y.flatten()**2 - A_y.flatten()**2))

    )
    u_h[:, :, 0] = value[:, np.newaxis]  

    return u_h


u_0_max = initial_value_not_equal_zero_no_symmetry(L_max, Q, N_max) #Using the non-symmetric initial value
#u_0_max = initial_value_not_equal_zero(L_max, Q, N_max)

u_0_iter = {}
for L in L_list:
    u_0_iter[L] = {}
    for n_iter in N_list:
        #u_0_iter[L][n_iter] = initial_value_not_equal_zero(L, Q, n_iter)
        u_0_iter[L][n_iter] = initial_value_not_equal_zero_no_symmetry(L, Q, n_iter)

print("Calculation initial value done")

def error_of_Q_paths(L_list, L_max, N_max, N_list, q, inverse_matrix_L_max_N_max, inverse_matrix_iter, T, Z, Y, Q):
    #Calculating the squared L^2-error for Q paths at the same time
    def solution_FVS_max(L_max, q, N_max, inverse_matrix_L_max_N_max, Z, Q): #Calculation of the solution of FVS for L_max and N_max
        
        u_h = u_0_max
        
        for k in range(N_max):
            for p in range(Q):
                u_h[:, p, k + 1] = (inverse_matrix_L_max_N_max @ ((4 / L_max**2) * (u_h[:, p, k] + np.sqrt(1 + u_h[:, p, k]**2) * Z[Q * q + p, k])))
            
        return u_h[:, :, N_max]

    def solution_FVS_iter(L, q, n_iter, inverse_matrix_iter, Y, Q): #Calculation of the solution of FVS for L in L_list and n_iter in N_list
 
        u_h = u_0_iter[L][n_iter]
        for k in range(n_iter):  
            for p in range(Q):
                u_h[:, p, k + 1] = (inverse_matrix_iter[L][n_iter] @ ((4 / L**2) * (u_h[:, p, k] + np.sqrt(1 + u_h[:, p, k]**2) * Y[n_iter][Q * q + p][k])))
    
        return u_h[:, :, n_iter] 

    U_max = solution_FVS_max(L_max, q, N_max, inverse_matrix_L_max_N_max, Z, Q) #solution of FVS for L_max and N_max
    U_appr = {} #collection of solutions of FVS for L in L_list and N in N_list
    U_appr_expanded = {} #expansion of the matrix-size of U_appr so that U_appr and U_max have the same size
    U_norm = {}
    
    for L in L_list:
        L_lcm = np.lcm(L_max, L)
        U_appr[L] = {}
        U_norm[L] = {}
        U_appr_expanded[L] = {}
        for n_iter in N_list:
            U_appr[L][n_iter] = solution_FVS_iter(L, q, n_iter, inverse_matrix_iter, Y, Q)
            U_appr_expanded[L][n_iter] = np.zeros((L_lcm**2, Q))
            for n in range (L**2):
                for k in range (L_lcm // L):
                    for l in range (L_lcm // L):
                        U_appr_expanded[L][n_iter][(k + (n // L) * (L_lcm // L)) * L_lcm + l + (n % L) * (L_lcm // L), :] = U_appr[L][n_iter][n, :]
                        # conversion formula to the bigger size L_lcm**2
            U_max_expanded = np.zeros((L_lcm**2, Q)) #expansion of the size so that U_appr and U_max have the same size
            for n in range (L_max**2):
                for k in range (L_lcm // L_max):
                    for l in range (L_lcm // L_max):
                        U_max_expanded[(k + (n // L_max) * (L_lcm // L_max)) * L_lcm + l + (n % L_max) * (L_lcm // L_max), :] = U_max[n, :]
                        # conversion formula to the bigger size L_lcm**2

            U_norm[L][n_iter] = (4 / L_lcm**2) * np.sum((U_appr_expanded[L][n_iter] - U_max_expanded)**2)

    return U_norm

Error = {}
for q in range(N_p // Q): 
    Error[q] = error_of_Q_paths(L_list, L_max, N_max, N_list, q, inverse_matrix_L_max_N_max, inverse_matrix_iter, T, Z, Y, Q)
    # Sum of the squared L^2-error for the paths from q*Q+1 to (q+1)*Q
    if q % (N_p // (10*Q)) == (N_p // (10*Q)) - 1: 
        print(f"Calculation for the first {(q+1)*Q} paths done")
        #This gives you info for each 10th of the whole calculation that this 10th of the whole calculation is done

total_norms = {} #collection of approximation of expectation of squared L^2-error for L in L_list and n_iter in N_list with Monte Carlo method

for L in L_list:
    total_norms[L] = {}
    for n_iter in N_list:
        total_norms[L][n_iter] = 0
        for q in range(N_p // Q):
            total_norms[L][n_iter] += Error[q][L][n_iter]
        total_norms[L][n_iter] /= N_p
        print(f"T = {T}, {N_p} paths, N_max = {N_max}, {n_iter} iterations, L_max = {L_max}, L = {L}: {total_norms[L][n_iter]}")

###


# In[ ]:




