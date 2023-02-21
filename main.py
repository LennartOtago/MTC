import numpy as np
import numpy.random as rd
from numpy.fft import fft2, ifft2
#initialize first samples
number_samples = 100
gammas = np.zeros(number_samples)
etas = np.zeros(number_samples)
accept_ratio = np.zeros(number_samples)

gammas[0] = 0.218
etas[0] = 5.15e-5


#draw new sample with normal prob

std_eta = 17.28e-7
std_gamma = 2.34e-3

for n in range(number_samples-1):
    gammas[n+1] = rd.normal(gammas[n], std_gamma)
    etas[n+1] = rd.normal(etas[n], std_eta)

#calculate acceptance ratio






#stencil
stencil = [[0, -1, 0],[-1, 4, -1],[0, -1, 0]]


#build data structure
siz = 9
img_siz = int(np.sqrt(siz))
numb_neig = 4

neigbours = np.zeros((siz,numb_neig))

for n in range(siz):

    #right neigbour

    if (n + 1) % img_siz == 0:
        neigbours[n, 0] = n - img_siz + 1
    else:
        neigbours[n,0] = n + 1
    #left neighbour
    if (n-1)% img_siz == img_siz-1:
        neigbours[n, 1] = n + img_siz - 1
    else:
        neigbours[n,1] = n - 1

    #up neigbour
    neigbours[n,2] = (n - img_siz) % siz

    #down neigbour
    neigbours[n,3] = (n + img_siz )% siz


L = np.zeros((siz,siz)) + numb_neig * np.identity(siz)


#build lalplacian matrix
for i in range(siz):
    for n in range(numb_neig):
        L[i,int(neigbours[i,n])] = -1
        L[int(neigbours[i,n]),i] = -1

#create one fourier basis vector
F_basM = np.zeros((img_siz,img_siz), dtype = 'csingle' )
for m in range(img_siz):
    for i in range(img_siz):
        for l in range(img_siz):
            for k in range(img_siz):
                #G[l,k] F[n,m,i]

                F_basM[m,i] = 1 * np.exp(-2j * np.pi * (m * 2 + i * 2) / img_siz) #/ img_siz
F_bas = F_basM.flatten()
F_bas2 = np.zeros(siz, dtype = 'csingle' )
for m in range(siz):
               F_bas2[m] = np.exp(-2j * np.pi * (m +1) / img_siz)

F_bas3 = np.zeros(siz, dtype = 'csingle' )
for m in range(siz):
               F_bas3[m] = np.exp(-2j * np.pi * (m +2) / img_siz)

#create fourier matrix for each pixel
F = np.zeros((siz,img_siz,img_siz), dtype = 'csingle')
W = np.zeros((siz,siz), dtype = 'csingle')

n = 0
for n in range(siz):

        G_vec = np.zeros(siz)
        G_vec[n] = numb_neig

        for x in range(numb_neig):
            G_vec[int(neigbours[n,x])] = -1
        G = np.reshape(G_vec, (img_siz, img_siz))
        print(G_vec)
        #FFT = np.zeros((img_siz, img_siz))
        for l in range(img_siz):
            for k in range(img_siz):
                for m in range(img_siz):
                    for i in range(img_siz):

                        #G[l,k] F[n,m,i]

                        F[n,m,i] =  np.exp(-2j * np.pi * (m * l + i * k + n) / img_siz) #/ img_siz

        Four_vec = F[n].flatten()
        W[n] = G_vec * Four_vec


u,s,vh = np.linalg.svd(L)

V = np.zeros((siz,siz), dtype = 'csingle')
for n in range(siz):
    V[:,n] = F[n].T.flatten()

v_0 = np.diag(V) * np.identity(siz)


Z = np.matmul(L, np.diag(V)).real

Z2 = np.matmul(L, V[:,3]).real

eig_val, eig_vec = np.linalg.eig(L)

print('debugggg')