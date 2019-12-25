#!/usr/bin/env python
# coding: utf-8
import numpy as np

'''
Inputs:
    x : training patterns (num_samples * n_d),
    y : training targets (num_samples * n_k),
    ker : kernel type ('lin', 'poly', 'rbf'),
    C : cost parameter,
    par : kernel parameter (see function 'kernelmatrix'),
    tol : tolerance.
Outputs:
    Beta
'''

def msvr(x, y, ker, C, epsi, par, tol):
    n_m = np.shape(x)[0]   #num of samples
    n_d = np.shape(x)[1]   #input data dimensionality
    n_k = np.shape(y)[1]   #output data dimensionality (output variables)
    
    #build the kernel matrix on the labeled samples
    H = kernelmatrix(ker, x, x, par)
    
    #create martix for regression parameters
    Beta = np.zeros((n_m, n_k))
    
    #E = prediction error per output (n_m * n_k)
    E = y - np.dot(H, Beta)
    #RSE
    u = np.sqrt(np.sum(E**2,1,keepdims=True))
    
    #RMSE
    RMSE = []
    RMSE_0 = np.sqrt(np.mean(u**2))
    RMSE.append(RMSE_0) 
    
    #points for which prediction error is larger than epsilon
    i1 = np.where(u>epsi)[0]
    
    #set initial values of alphas a (n_m * 1)
    a = 2 * C * (u - epsi) / u
    
    #L (n_m * 1)
    L = np.zeros(u.shape)
    
    # we modify only entries for which  u > epsi. with the sq slack
    L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2
    
    #Lp is the quantity to minimize (sq norm of parameters + slacks)    
    Lp = []
    BetaH = np.dot(np.dot(Beta.T, H), Beta)
    Lp_0 = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
    Lp.append(Lp_0)
    
    eta = 1
    k = 1
    hacer = 1
    val = 1
    
    while(hacer):
        Beta_a = Beta.copy()
        E_a = E.copy()
        u_a = u.copy()
        i1_a = i1.copy()
        
        M1 = H[i1][:,i1] + np.diagflat(1/a[i1]) + 1e-10 * np.eye(len(a[i1]))
        
        #compute betas
        sal1 = np.dot(np.linalg.pinv(M1),y[i1])  #求逆or广义逆（M-P逆）无法保证M1一定是可逆的？
#         sal1 = np.dot(np.linalg.inv(M1),y[i1])
        
        eta = 1
        Beta = np.zeros(Beta.shape)
        Beta[i1] = sal1.copy()
        
        #error
        E = y - np.dot(H, Beta)
        #RSE
        u = np.sqrt(np.sum(E**2,1)).reshape(n_m,1)
        i1 = np.where(u>=epsi)[0]
        
        L = np.zeros(u.shape)
        L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2
        
        #%recompute the loss function
        BetaH = np.dot(np.dot(Beta.T, H), Beta)
        Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
        Lp.append(Lp_k)
        
        #Loop where we keep alphas and modify betas
        while(Lp[k] > Lp[k-1]):
            eta = eta/10
            i1 = i1_a.copy()
            
            Beta = np.zeros(Beta.shape)
            #%the new betas are a combination of the current (sal1) 
            #and of the previous iteration (Beta_a)
            Beta[i1] = eta*sal1 + (1-eta)*Beta_a[i1]
            
            E = y - np.dot(H, Beta)
            u = np.sqrt(np.sum(E**2,1)).reshape(n_m,1)

            i1 = np.where(u>=epsi)[0]
            
            L = np.zeros(u.shape)
            L[i1] = u[i1]**2 - 2 * epsi * u[i1] + epsi**2
            BetaH = np.dot(np.dot(Beta.T, H), Beta)
            Lp_k = np.sum(np.diag(BetaH), 0) / 2 + C * np.sum(L)/2
            Lp[k] = Lp_k
            
            #stopping criterion 1
            if(eta < 1e-16):
                Lp[k] = Lp[k-1]- 1e-15
                Beta = Beta_a.copy()
                
                u = u_a.copy()
                i1 = i1_a.copy()
                
                hacer = 0
        
        #here we modify the alphas and keep betas
        a_a = a.copy()
        a = 2 * C * (u - epsi) / u
        
        RMSE_k = np.sqrt(np.mean(u**2))
        RMSE.append(RMSE_k)
        
        if((Lp[k-1]-Lp[k])/Lp[k-1] < tol):
            hacer = 0
            
        k = k + 1
        
        #stopping criterion #algorithm does not converge. (val = -1)
        if(len(i1) == 0):
            hacer = 0
            Beta = np.zeros(Beta.shape)
            val = -1
            
    NSV = len(i1)
    
    return Beta

'''
KERNELMATRIX

Builds a kernel from training and test data matrices. 

Inputs: 
    ker: {'lin' 'poly' 'rbf'}
    X: Xtest (num_test * n_d)
    X2: Xtrain (num_train * n_d)
    parameter: 
       width of the RBF kernel
       bias in the linear and polinomial kernel 
       degree in the polynomial kernel

Output:
    K: kernel matrix
'''

def kernelmatrix(ker, X, X2, p=0):

    X = X.T
    X2 = X2.T

    if(ker == 'lin'):
        tmp1, XX2_norm, tmp2 = np.linalg.svd(np.dot(X.T,X2))
        XX2_norm = np.max(XX2_norm)
        K = np.dot(X.T,X2)/XX2_norm
    
    elif(ker == 'poly'):
        tmp1, XX2_norm, tmp2 = np.linalg.svd(np.dot(X.T,X2))
        XX2_norm = np.max(XX2_norm)
        K = (np.dot(X.T,X2)/XX2_norm*p[0] + p[1]) ** p[2]
    
    elif(ker == 'rbf'):
        n1sq = np.sum(X**2,0,keepdims=True)
        n1 = X.shape[1]
        
        if(n1 == 1):        #just one feature
            N1 = X.shape[0]
            N2 = X2.shape[0]
            D = np.zeros((N1,N2))
            for i in range(0,N1):
                D[i] = (X2 - np.dot(np.ones((N2,1)),X[i].reshape(1,-1))).T * (X2 - np.dot(np.ones((N2,1)),X[i].reshape(1,-1))).T
        else:
            n2sq = np.sum(X2**2,0,keepdims=True)
            n2 = X2.shape[1]
            D = (np.dot(np.ones((n2,1)),n1sq)).T + np.dot(np.ones((n1,1)),n2sq) - 2*np.dot(X.T, X2)
        
        K = np.exp((-D**2)/(2*p**2))
        
    else:
        print("no such kernel")
        K = 0
        
    return K