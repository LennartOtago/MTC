import numpy as np
import numpy.random as rd
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

#create fourier matrix for each pixel
F = np.zeros((siz,img_siz,img_siz), dtype = 'csingle')

n = 0
for n in range(siz):

        G = np.zeros(siz)
        G[n] = numb_neig
        for x in range(numb_neig):
            G[int(neigbours[n,x])] = -1
        G = np.reshape(G, (img_siz, img_siz))

        #FFT = np.zeros((img_siz, img_siz))

        for m in range(img_siz):
            for i in range(img_siz):
                for l in range(img_siz):
                    for k in range(img_siz):
                        #G[l,k]
                        F[n,m,i] = F[n,m,i] + 1 * np.exp(-2j * np.pi * (m * l + i * k)/ img_siz) / img_siz



W = np.ndarray(shape = (siz,siz), dtype = 'csingle')
W_trans = np.zeros((siz,siz))
for m in range(siz):
    for n in range(siz):
        W[m,n] =  np.exp( 2j * np.pi * m * n / siz) / np.sqrt(siz)

W_trans =   W.T.conj() / np.sqrt(siz)


v_0 = F[0].T.flatten().real

Z = np.matmul(L,v_0).real

Z2 = np.matmul(L,W[:,1]).real
c = W[:,1].real
print('debugggg')