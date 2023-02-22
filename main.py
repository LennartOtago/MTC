import numpy as np
import numpy.random as rd
from functions import *
from numpy.fft import fft2, ifft2
import matplotlib.image as mpimg
from numpy.random import normal
import matplotlib.pyplot as plt

#build data structure
# siz = 9
# img_siz = int(np.sqrt(siz))
# numb_neig = 4
# neigbors = gen_neigborhood(siz, numb_neig)
# L,D,Q = gen_and_diag_L(siz, neigbors, numb_neig)

# X = np.zeros((500,256,256))
#
# for x in range(500):
#     X[x] = sample_laplacian(256)

# intialize data
gray_img = mpimg.imread('jupiter1.tif')

org_img = np.array(gray_img)
Y = fft2(org_img)

# get psf from satellite
xpos = 234
ypos = 85  # Pixel at centre of satellite
sat_img_org = org_img[ypos - 16: ypos + 16, xpos - 16:xpos + 16]
sat_img = sat_img_org / (sum(sum(sat_img_org)))
sat_img[sat_img < 0.05*np.max(sat_img)] = 0
A = fft2(sat_img, (256,256))

#stencil for laplacian
stencil = [[0, -1, 0],[-1, 4, -1],[0, -1, 0]]
L_fft = fft2(stencil,(256,256))

#initialize first samples
number_samples = 100
gammas = np.zeros(number_samples)
etas = np.zeros(number_samples)
accept_ratio = np.zeros(number_samples-1)

gammas[0] = 0.218
etas[0] = 5.15e-5
gammas[-1] = 0.218
etas[-1] = 5.15e-5

#draw new sample with normal prob

std_eta = 17.28e-7
std_gamma = 2.34e-3
k = 0
for n in range(number_samples-1):

    #sample new hyperparmeters
    gammas[n+1] = rd.normal(gammas[n], std_gamma)
    etas[n+1] = rd.normal(etas[n], std_eta)

    #calculate log determinant g and f
    lam_old = etas[n]/gammas[n]
    f_old = f(Y,lam_old,L_fft,A).real
    g_old = g(A, lam_old,L_fft).real

    lam_new = etas[n+1] / gammas[n+1]
    f_new = f(Y, lam_new, L_fft, A).real
    g_new = g(A, lam_new, L_fft).real

    #calculate acceptance ratio

    ratio = (np.log(etas[n+1]) - np.log(etas[n])) * 256**2/2 - g_new/2 - gammas[n+1] * f_new / 2 + g_old/2 + gammas[n] * f_old /2 \
            + (gammas[n+1] - gammas[n])**2 - (gammas[n] - gammas[n-1])**2 + (etas[n+1] - etas[n])**2 - (etas[n] - etas[n-1])**2

    accept_ratio[n] =  abs(np.exp(ratio))
    while rd.uniform() > abs(np.exp(ratio)):
        print("new state rejected")
        k = k+1
        gammas[n + 1] = rd.normal(gammas[n], std_gamma)
        etas[n + 1] = rd.normal(etas[n], std_eta)

        lam_new = etas[n + 1] / gammas[n + 1]
        f_new = f(Y, lam_new, L_fft, A).real
        g_new = g(A, lam_new, L_fft).real
        ratio = (np.log(etas[n+1]) - np.log(etas[n])) * 256**2/2 - g_new/2 - gammas[n+1] * f_new / 2 + g_old/2 + gammas[n] * f_old /2 \
            #+ (gammas[n+1] - gammas[n])**2 - (gammas[n] - gammas[n-1])**2 + (etas[n+1] - etas[n])**2 - (etas[n] - etas[n-1])**2
        accept_ratio[n] =  abs(np.exp(ratio))


    #sample new image
    #v_rd = normal(0, 1, 256 ** 2).reshape((256, 256))
    #W = np.sqrt(gammas[n+1]) * np.conj(A) * fft2(v_rd) + np.sqrt(etas[n+1]) * fft2(sample_laplacian(256))

plt.figure()
plt.subplot(3,1,1)
plt.hist(gammas)
plt.subplot(3,1,2)
plt.hist(etas)
plt.subplot(3,1,3)
plt.hist(etas/gammas)
plt.show()


lam = np.logspace(-10,5,150)
y = np.array([(f(Y,lamba,L_fft,A)) for lamba in lam ])
fig2 = plt.figure()
ax = fig2.add_subplot()
plt.plot(lam,y.real)
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
#
# fig3 = plt.figure()
# ax = fig3.add_subplot()
# plt.plot(lam,[(g(A,lamba,L_fft)).real for lamba in lam ] )
# ax.set_xscale('log')
# #ax.set_yscale('log')
# plt.show()

print('debugggg')