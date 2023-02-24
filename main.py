import numpy as np
import numpy.random as rd
from functions import *

from numpy.fft import fft2, ifft2
import matplotlib.image as mpimg
from numpy.random import normal,  multivariate_normal
import matplotlib.pyplot as plt

#build data structure
# siz = 9
# img_siz = int(np.sqrt(siz))
# numb_neig = 4
# neigbors = gen_neigborhood(siz, numb_neig)
# L,D,Q = gen_and_diag_L(siz, neigbors, numb_neig)

# X = np.zeros((1000,256,256))
#
# for x in range(1000):
#     print(x)
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
number_samples = 5000
kappa = 1000
gammas = np.zeros(number_samples)
etas = np.zeros(number_samples)
accept_ratio = np.zeros(number_samples-1)
res_img = np.zeros((number_samples,256,256))

gammas[0] = 0.218
etas[0] = 5.15e-5
gammas[-1] = 0.218
etas[-1] = 5.15e-5

#intial image
v_rd = normal(0, 1, 256 ** 2).reshape((256, 256))
W = np.sqrt(gammas[0]) * np.conj(A) * fft2(v_rd) + np.sqrt(etas[0]) * fft2(sample_laplacian(256))
IMG_old = (gammas[0] * Y * np.conj(A) + W) / (
        etas[0] * abs(L_fft) + gammas[0] * abs(A) ** 2)
res_img[0] = ifft2(
    (gammas[0] * Y * np.conj(A) + W) / (etas[0] * abs(L_fft) + gammas[0] * abs(A) ** 2)).real




#draw new sample with normal prob

std_eta = 17.28e-7/2
std_gamma = 2.34e-3/2
k = 0
for n in range(number_samples-1):



    #sample new hyperparmeters
    gammas[n+1] = rd.normal(gammas[n], std_gamma)
    etas[n+1] = rd.normal(etas[n], std_eta)

    #calculate log determinant g and f
    lam_old = etas[n]/gammas[n]
    f_old = f(Y,lam_old,L_fft,A)
    g_old = g(A, lam_old,L_fft)

    lam_new = etas[n+1] / gammas[n+1]
    f_new = f(Y, lam_new, L_fft, A)
    g_new = g(A, lam_new, L_fft)

    #calculate acceptance ratio

    ratio = (np.log(etas[n+1]) - np.log(etas[n])) * 256**2/2 - g_new/2 - gammas[n+1] * f_new / 2 + g_old/2 + gammas[n] * f_old /2 \
            + (gammas[n] - gammas[n + 1]) * 1e-4 + (etas[n] - etas[n + 1]) * 1e-4

    accept_ratio[n] =  np.exp(ratio)
    while rd.uniform() > np.exp(ratio):
        print("new state rejected")
        k = k+1
        gammas[n + 1] = rd.normal(gammas[n], std_gamma)
        etas[n + 1] = rd.normal(etas[n], std_eta)

        lam_new = etas[n + 1] / gammas[n + 1]
        f_new = f(Y, lam_new, L_fft, A)
        g_new = g(A, lam_new, L_fft)
        ratio = (np.log(etas[n+1]) - np.log(etas[n])) * 256**2/2 - g_new/2 - gammas[n+1] * f_new / 2 + g_old/2 + gammas[n] * f_old /2 \
             + (gammas[n] -gammas[n+1] )*1e-4 + (etas[n] - etas[n+1]) *1e-4
        accept_ratio[n] =  np.exp(ratio)


    #sample new image
    # v_rd = normal(0, 1, 256 ** 2).reshape((256, 256))
    # W = np.sqrt(gammas[n+1]) * np.conj(A) * fft2(v_rd) + np.sqrt(etas[n+1]) * fft2(sample_laplacian(256))
    # IMG_NEW = (gammas[n+1] * Y * np.conj(A) + W) / (
    #         etas[n+1] * abs(L_fft) + gammas[n+1] * abs(A) ** 2)
    # res_img[n+1] = ifft2(
    #     (gammas[n+1] * Y * np.conj(A) + W) / (etas[n+1] * abs(L_fft) + gammas[n+1] * abs(A) ** 2)).real

    #accept new image
burn = 20

#calculate autocorrelation function

auto1 = auto_corr_fast(etas[burn::]/gammas[burn::],kappa)
auto3 = auto_corr(etas[burn::]/gammas[burn::],kappa)
tau = tau(etas[burn::]/gammas[burn::],kappa)

plt.figure()
ax = plt.subplot()
plt.plot(auto1)
ax.set_xlim([0,40])
ax.set_ylim([-0.2,1])
#plt.show()

import statsmodels.api as sm
nlags = number_samples-burn-1#int(number_samples/10)
auto_eta  = sm.tsa.acf(etas[burn::], nlags=nlags)
auto_gamma  = sm.tsa.acf(gammas[burn::], nlags=nlags)
auto_lam = sm.tsa.acf(etas[burn::]/gammas[burn::], nlags=nlags)
int_eta = 1 + 2 * sum(auto_eta)
int_gamma = 1 + 2 * sum(auto_gamma)
int_lam = 1 + 2 * sum(auto_lam)

plt.figure()
ax = plt.subplot()
plt.plot(np.linspace(0,nlags,nlags+1),auto_lam)
ax.set_ylim([-0.2, 1])
ax.set_xlim([0, 40])
plt.show()

plt.figure()
ax1 = plt.subplot(3,1,1)
plt.hist(gammas[burn::])
ax1.set_xlim([0.212, 0.226])
ax2 = plt.subplot(3,1,2)
plt.hist(etas[burn::]*1e5)
ax2.set_xlim([4.6, 5.6])
ax3 = plt.subplot(3,1,3)
plt.hist(etas[burn::]/gammas[burn::]*1e4)
ax3.set_xlim([2.2, 2.6])
plt.show()



# lam = np.logspace(-10,5,150)
# y = np.array([(f(Y,lamba,L_fft,A)) for lamba in lam ])
# fig2 = plt.figure()
# ax = fig2.add_subplot()
# plt.plot(lam,y.real)
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.show()
#
# fig3 = plt.figure()
# ax = fig3.add_subplot()
# plt.plot(lam,[(g(A,lamba,L_fft)).real for lamba in lam ] )
# ax.set_xscale('log')
# #ax.set_yscale('log')
# plt.show()


np.savetxt('samples.txt', np.vstack((etas,gammas)).T, header = 'etas \t gammas', fmt = '%.15f \t %.15f')

# plt.figure(4)
# plt.scatter(L_fft.real,L_fft.imag)
# plt.show()

print(np.mean(accept_ratio[accept_ratio < 1]))
print('debugggg')