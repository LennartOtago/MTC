import numpy as np
from numpy.random import normal,  multivariate_normal

def gen_and_diag_L(siz, neigbours, numb_neig):

    img_siz = int(np.sqrt(siz))
    L = np.zeros((siz, siz)) + numb_neig * np.identity(siz)

    # build lalplacian matrix
    for i in range(siz):
        for n in range(numb_neig):
            L[i, int(neigbours[i, n])] = -1
            L[int(neigbours[i, n]), i] = -1

    # create fourier matrix for each pixel
    F = np.zeros((siz, img_siz, img_siz), dtype='csingle')
    # W = np.zeros((siz,siz), dtype = 'csingle')

    n = 0

    for l in range(img_siz):
        for k in range(img_siz):
            # Storage = np.zeros((img_siz, img_siz), dtype='csingle')
            # G_vec = np.zeros(siz)
            # G_vec[n] = numb_neig
            # for x in range(numb_neig):
            #     G_vec[int(neigbours[n, x])] = -1
            # G = np.reshape(G_vec, (img_siz, img_siz))
            # print(G)
            # print(abs(fft2(G)))
            for m in range(img_siz):
                for i in range(img_siz):
                    # G[l,k] F[n,m,i]
                    #print(m * l + i * k)
                    F[n, m, i] = np.exp(-2j * np.pi * (m * l + i * k) / img_siz) / img_siz

            n = n + 1

    Q = np.zeros((siz, siz), dtype='csingle')
    for n in range(siz):
        Q[:, n] = F[n].flatten()

    # check if orthogonal bases
    QTQ = np.around(np.matmul(Q.T.conj(), Q).real)
    if np.allclose(QTQ,np.identity(siz)):
        print("Q (2D Fourier basis) are orthonormal")

    D = np.around(np.matmul(Q.conj().T, np.matmul(L, Q)).real)
    return L, D, Q

def gen_neigborhood(siz, numb_neig):
    img_siz = int(np.sqrt(siz))
    neigbors = np.zeros((siz, numb_neig))

    for n in range(siz):

        # right neigbour

        if (n + 1) % img_siz == 0:
            neigbors[n, 0] = n - img_siz + 1
        else:
            neigbors[n, 0] = n + 1
        # left neighbour
        if (n - 1) % img_siz == img_siz - 1:
            neigbors[n, 1] = n + img_siz - 1
        else:
            neigbors[n, 1] = n - 1

        # up neigbour
        neigbors[n, 2] = (n - img_siz) % siz

        # down neigbour
        neigbors[n, 3] = (n + img_siz) % siz
    return neigbors

def f(Y,lam,L,A):
    return sum(sum(  abs(Y)**2 * lam * abs(L)/ abs(A)**2  / (1 + lam * abs(L) / abs(A)**2) ) )  /256**2


def g(A,lam,L):
    return  sum(sum(np.log(abs(A)**2 ))) + sum(sum(np.log(1 + lam * abs(L)/abs(A)**2) ))#-(256**2)


def sample_laplacian(siz_of_img):
    # ***
    # sample from Laplacian matrix in this case with variance 4 and 0 mean
    # original L L_org = np.array([[ 0, -1, 0],[ -1, 4, -1],[ 0, -1, 0]])
    # ***
    mean = (0, 0)
    cov = [[1, 0], [0, 1]]

    v = np.zeros((siz_of_img, siz_of_img))
    # top to bottom first row
    for i in range(0, siz_of_img):
        rand_num = np.array([-1, 1]) * normal(0, 1,2)#multivariate_normal(mean, cov)#normal(0, 1)#np.sqrt(2))#(1 / np.sqrt(2)) *
        v[0, i], v[-1, i] = [v[0, i], v[-1, i]] + np.array(rand_num)

    for j in range(0, siz_of_img):
        for i in range(0, siz_of_img - 1):
            rand_num =  np.array([-1, 1]) *normal(0, 1,2)# multivariate_normal(mean, cov)#normal(0, 1)#(1 / np.sqrt(2)) *
            v[j, i], v[j, i + 1] = [v[j, i], v[j, i + 1]] + np.array(rand_num)

    # all normal up and down neighbours

    for j in range(0, siz_of_img - 1):
        for i in range(0, siz_of_img):
            rand_num =  np.array([-1, 1]) *normal(0, 1,2)# multivariate_normal(mean, cov)#normal(0,1)#np.sqrt(2))#(1 / np.sqrt(2)) *
            v[j, i], v[j + 1, i] = [v[j, i], v[j + 1, i]] + np.array(rand_num)

    # all left right boundaries neighbours

    for i in range(0, siz_of_img):
        rand_num =  np.array([-1, 1]) * normal(0, 1,2)#multivariate_normal(mean, cov)#normal(0,1)# np.sqrt(2))#(1 / np.sqrt(2))
        v[i, 0], v[i, -1] = [v[i, 0], v[i, -1]] + np.array(rand_num)

    return v

'''
Calculates the autocorrelation curve of M, then returns the integrated 
autocorrelation time (IAC).
'''
def tau(M,kappa):
#   autocorr is the UNintegrated autocorrelation curve
    autocorr = auto_corr_fast(M,kappa)
#   tau = 1 + 2*sum(G)
    return 1 + 2*np.sum(autocorr), autocorr

'''
This is the intuitive and naive way to calculate autocorrelations. See
auto_corr_fast(...) instead.
'''
def auto_corr(M,kappa):
#   The autocorrelation has to be truncated at some point so there are enough
#   data points constructing each lag. Let kappa be the cutoff
    auto_corr = np.zeros(kappa-1)
    mu = np.mean(M)
    for s in range(1,kappa-1):
        auto_corr[s] = np.mean( (M[:-s]-mu) * (M[s:]-mu) ) / np.var(M)
    return auto_corr

'''
The bruteforce way to calculate autocorrelation curves is defined in 
auto_corr(M). The correlation is computed for an array against itself, and 
then the indices of one copy of the array are shifted by one and the 
procedure is repeated. This is a typical "convolution-style" approach. 
An incredibly faster method is to Fourier transform the array first, since 
convolutions in Fourier space is simple multiplications. This is the approach
in auto_corr_fast(...)
'''
def auto_corr_fast(M,kappa):
#   The autocorrelation has to be truncated at some point so there are enough
#   data points constructing each lag. Let kappa be the cutoff
    M = M - np.mean(M)
    N = len(M)
    fvi = np.fft.fft(M, n=2*N)
#   G is the autocorrelation curve
    G = np.real( np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
    G /= N - np.arange(N); G /= G[0]
    G = G[:kappa]
    return G