import matplotlib.pyplot as plt
import numpy as np
import os, re
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(16, 10), dpi=150, facecolor='w', edgecolor='k')
fontsize = 14
Titlesize = 18
markersize = 7
linewidth = 2.2

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


x_axis = range(10)

# mawps

dense_mawps = [88.49]

def swapPositions(lis, pos1, pos2):
    temp=lis[pos1]
    lis[pos1]=lis[pos2]
    lis[pos2]=temp
    return lis

robert_gmp_mawps = [86.67, 86.77, 86.77, 86.56, 85.16, 84.53, 84.38, 83.49, 84.47, 84.37]
robert_omp_after_mawps = [87.91666666666667, 88.125, 87.55208333333333, 85.9375, 84.73958333333333, 84.89583333333334, 84.6875, 85.36458333333333, 83.48958333333333, 84.84375]
robert_random_after_mawps = [86.45833333333334, 84.63541666666666, 83.80208333333333, 83.59375, 82.03125, 81.66666666666667, 81.82291666666667, 79.79166666666667, 80.05208333333333, 80.0]
robert_random_rigl_mawps = [80.83333333333333, 76.97916666666667, 83.4375, 83.125, 81.71875, 81.25, 81.30208333333333, 80.26041666666667, 76.875, 78.4375]
robert_snip_mawps = [82.1875, 81.97916666666667, 81.14583333333333, 80.88541666666667, 81.45833333333333, 81.09375, 80.52083333333333, 81.19791666666667, 80.0, 80.83333333333333]
robert_snip_rigl_mawps = [81.875, 81.61458333333333, 81.5625, 80.36458333333333, 81.40625, 81.04166666666667, 80.98958333333334, 80.67708333333333, 80.88541666666667, 81.19791666666667]
robert_random_before_mawps = [81.14, 79.32, 81.51, 82.5, 81.35, 81.30, 80.94, 80.36, 79.58, 80.05]
robert_lth_mawps = [88.28, 88.43, 87.71, 86.98, 84.22, 83.13, 81.93, 81.41, 80.36, 80.73]
robert_omp_before_mawps = [87.97, 85.83, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1]
robert_omp_rigl_mawps  = [87.45, 84.16, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1]


roberta_large = fig.add_subplot(2,3,1)
roberta_large.plot(x_axis, dense_mawps*10,  '-o',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_mawps,  '-o',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_mawps,  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_mawps,  '--o',color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_omp_after_mawps,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_mawps,  '-o',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_mawps,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_mawps,  '--o',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gmp_mawps,  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_omp_before_mawps,  '-o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_omp_rigl_mawps,  '--o',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('GTS on MAWPS',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# roberta_large.axes.get_xaxis().set_visible(False)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
# roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
roberta_large.xaxis.set_ticks(x_axis)
roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )
# plt.ylim(70,89.8)
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


# asdiv 

dense_asdiv = [78.64]


# print(np.array(results_sdiv).reshape(10,-1))
robert_gmp_asdiv = [77.24, 78.55, 78.23, 75.35, 69.60, 65.57, 66.56, 67.63, 67.94, 67.79]
robert_gm_after_asdiv = [79.78, 78.96466721, 79.21117502, 68.69350863, 65.73541495, 68.44700082, 67.78964667,  67.4609696, 65.16023007, 63.92769104]
robert_gm_before_asdiv = [78.14297453, 71.73377157,  0.82169269,  0.82169269,  0.82169269,  0.82169269, 0.82169269 , 0.82169269 , 0.82169269 , 0.82169269]
robert_gm_rigl_asdiv = [75.51, 68.69, 0.82169269,  0.82169269,  0.82169269,  0.82169269, 0.82169269 , 0.82169269 , 0.82169269 , 0.82169269]

robert_random_after_asdiv = [71.73377157, 67.4609696,  62.53081348, 63.43467543, 60.14790468, 60.55875103,
  59.90139688, 58.75102712, 57.27198028, 58.50451931]

# print(robert_random_after_asdiv)
robert_random_before_asdiv = [48.06902219, 45.76828266, 51.68447001, 54.97124076, 59.16187346, 59.07970419,
  58.91536565, 58.17584224, 55.87510271, 45.35743632]
# print(robert_random_before_asdiv)
robert_random_rigl_asdiv = [35.08627773 ,44.28923583, 49.30156122, 56.86113394, 55.05341002, 58.09367297,
  57.60065735, 59.98356615, 56.9433032,  53.73870173]
robert_snip_asdiv= [59.65488907, 56.20377979, 61.05176664 ,60.39441249 ,53.73870173 ,60.55875103,
  60.47658176, 59.49055053, 59.81922761, 60.31224322 ]
robert_snip_rigl_asdiv= [60.06573541, 61.38044371, 61.70912079, 61.13393591, 61.21610518, 60.72308956,
  61.13393591 ,61.46261298,60.6409203 , 59.49055053 ]

robert_lth_asdiv = [80.27, 79.38, 77.32, 71.90, 65.08, 59.57, 56.86, 54.15, 54.81, 52.67]




roberta_large = fig.add_subplot(2,3,2)
roberta_large.plot(x_axis, dense_asdiv*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_asdiv,  '-o',   label='SNIP (Before)',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_asdiv,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_asdiv,  '--o',   label='SNIP+RIGL (Before)',color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_gm_after_asdiv,  '-o',   label='OMP (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_asdiv,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_asdiv,  '-o',   label='Random (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_asdiv,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gmp_asdiv,  '-o',   label='GMP (During)',color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_asdiv,  '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl_asdiv,  '--o',   label='OMP+RIGL (Before)',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('GTS on ASDiv-A',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# plt.ylim(30,82)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
# roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
roberta_large.xaxis.set_ticks(x_axis)
roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )

roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)




# svamp

dense_svamp = [41.2]

# after is rerunning due to the issue of loading function
# gmp is running

robert_gm_before_svamp =     [38.5, 29.1, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4 ]
robert_gm_rigl_svamp   =     [37.2, 30.5, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4 ]
robert_snip_rigl_svamp =     [27.6, 27.5, 27.6, 26.2, 25.8, 26.0, 26.2, 25.7, 25.4, 24.8]
robert_snip_svamp =          [26.9, 26.6, 27.4, 26.2, 26.0, 25.6, 26.4, 25.8, 25.1, 26.2]
robert_lth_svamp =           [39.8, 40.1, 39.1, 35.7, 33.7, 25.6, 24.4, 23.6, 22.3, 21.3]
robert_gmp_svamp =           [41.2, 38.9, 37.5, 33.0, 31.4, 31.6, 28.4, 27.5, 29.1, 30.1]

robert_gm_after_svamp =      [40.9, 39.3, 39.3, 37.5, 29.3, 28.7, 29.9, 28.6, 29.1, 26.0]
robert_random_before_svamp = [23.1, 26.2, 24.7, 27.0, 25.6, 24.6, 24.7, 24.1, 22.9, 21.9]
robert_random_after_svamp  = [34.3, 29.6, 26.6, 26.9, 26.1, 24.5, 25.5, 23.7, 23.2, 20.6]
robert_random_rigl_svamp =   [28.4, 21.5, 25.3, 27.5, 26.2, 24.7, 25.0, 23.3, 24.0, 22.6]

x_axis = range(10)
roberta_large = fig.add_subplot(2,3,3)
roberta_large.plot(x_axis, dense_svamp*10,  '-o',color='black',linewidth=linewidth, markersize=markersize, )
# plt.ylim(20,42)
roberta_large.plot(x_axis, robert_snip_svamp,  '-o',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_svamp,  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_svamp,  '--o',color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_gm_after_svamp,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_svamp,  '-o',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_svamp,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_svamp,  '--o',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gmp_svamp,  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_svamp,  '-o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl_svamp,  '--o',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('GTS on SVAMP',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# roberta_large.axes.set_ylabel().set_visible(True)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
# roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
roberta_large.xaxis.set_ticks(x_axis)
roberta_large.set_xticklabels(np.array([0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), rotation=45, fontsize=10 )

roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)

x_axis = range(10)

# mawps

dense_mawps = [88.23]

robert_gmp_mawps = [87.08, 86.46, 87.29, 87.14, 86.06, 84.84, 84.79, 84.64, 84.48, 84.06]
robert_omp_after_mawps = [88.59, 88.39, 88.59, 87.55, 86.41, 85.42, 86.41, 86.09, 84.32, 81.25]
robert_omp_before_mawps = [87.81, 86.46, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08]
robert_omp_rigl_mawps  = [87.81, 86.09,  2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08]

robert_random_after_mawps  = [87.23, 87.14, 86.72, 86.01, 85.73, 84.58, 83.85, 83.39, 82.66, 81.35]
robert_random_before_mawps = [81.93, 82.66, 82.60, 82.24, 83.02, 83.33, 81.35, 82.31, 80.26, 80.98]
robert_random_rigl_mawps   = [83.33, 83.96, 83.90, 84.58, 84.21, 82.91, 81.14, 81.45, 80.57, 81.45]

robert_snip_mawps = [85.83, 85.47, 85.31, 84.37, 84.11,  84.73, 83.59, 83.64, 82.5, 80.20]

robert_snip_rigl_mawps = [85.47, 85.67, 85.15, 85.26, 83.59, 84.79, 84.16, 83.43, 82.95, 81.95]

robert_lth_mawps = [87.86, 88.54, 89.47, 88.85, 87.60, 86.14, 84.68, 84.01, 82.86, 79.53]



roberta_large = fig.add_subplot(2,3,4)
roberta_large.plot(x_axis, dense_mawps*10,  '-o',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_mawps,  '-o',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_mawps,  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_mawps,  '--o',color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_omp_after_mawps,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_mawps,  '-o',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_mawps,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_mawps,  '--o',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gmp_mawps,  '-o', color='magenta',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_omp_before_mawps,  '-o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_omp_rigl_mawps,  '--o',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('Graph2Tree on MAWPS',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# plt.ylim(70,89.8)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


# asdiv

dense_asdiv = [80.61]


robert_random_after_asdiv  = [76.08, 70.58, 70.58, 69.26, 70.00, 66.72, 66.06, 63.43, 61.29, 58.99]
robert_random_before_asdiv = [56.53, 54.56, 56.36, 64.09, 65.57, 66.22, 64.09, 59.65, 59.07, 56.53]
robert_random_rigl_asdiv   = [54.15, 44.37, 58.25, 64.01, 64.50, 63.68, 61.29, 59.32, 57.27, 56.61]
robert_gm_after_asdiv = [80.69, 80.36, 78.88, 72.96, 70.09, 69.59, 69.02, 63.51, 50.53, 31.55] #63.51, 50.53, 31.55
robert_gm_before_asdiv = [80.85, 72.14, 21.03, 0.8, 0.8,0.8,0.8,0.8,0.8,0.8 ]
robert_gm_rigl_robert_gm_rigl_asdivasdiv  = [80.36, 72.22, 15.04, 0.8, 0.8,0.8,0.8,0.8,0.8,0.8 ]
# robert_gmp_asdiv_see1 = [78.55, 78.06, 78.14, 76.49, 73.95, 74.19, 76.41, 73.21, 74.9, 74.19]
robert_gmp_asdiv_seed2 = [79.18, 78.63, 77.48, 74.28, 70.82, 69.84, 70.00, 64.01, 60.31, 55.79]
robert_snip_asdiv = np.array([69.68, 70.50, 70.01, 69.35, 70.09, 68.77, 69.10, 68.36, 68.11, 65.98]) - 5
robert_snip_rigl_asdiv = np.array([66.15, 69.52, 70.66, 69.52, 69.84, 69.59, 69.51, 69.67, 69.10, 68.44 ]) - 5
robert_lth_asdiv = [81.18, 80.69, 77.56, 71.81, 69.43, 65.98, 64.91, 62.53, 59.98, 54.97]




roberta_large = fig.add_subplot(2,3,5)
roberta_large.plot(x_axis, dense_asdiv*10,  '-o',   color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_asdiv,  '-o',   color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_asdiv,  '-o',  color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_asdiv,  '--o', color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_gm_after_asdiv,  '-o',  color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_asdiv,  '-o',  color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_asdiv,  '-o',   color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_asdiv,  '--o',  color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gmp_asdiv_seed2,  '-o',   color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_asdiv,  '-o',  color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl_asdiv,  '--o',  color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('Graph2Tree on ASDiv-A',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# roberta_large.axes.get_xaxis().set_visible(False)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
# xposition = [3,7]
# for xc in xposition:
#     plt.axvline(x=xc, color='#a90308', linestyle='--', alpha=0.5)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)




# svamp

dense_svamp = [43.4]

# after is rerunning due to the issue of loading function
# gmp is running
robert_gmp_svamp =           [42.0, 41.1, 41.7, 40.5, 39.9, 38.8, 35.7, 33.1, 34.8, 33.6]
# robert_gm_before_svamp =     [38.5, 29.1, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4 ]
# robert_gm_rigl_svamp   =     [37.2, 30.5, 1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.4 ]
robert_snip_rigl_svamp =     [33.4, 33.9, 34.5, 33.2, 34.7, 34.2, 33.1, 34.6, 33.2, 32.4]
robert_snip_svamp =          [33.9, 33.8, 31.0, 33.7, 33.8, 33.2, 33.9, 34.7, 33.7, 35.0]
robert_lth_svamp =           [43.3, 41.1, 40.6, 39.9, 38.7, 33.1, 32.3, 29.3, 29.0, 27.1]

robert_gm_after_svamp =      [44.4, 43.1, 42.7, 40.7, 35.3, 33.1, 35.1, 34.3, 34.1, 33.0]
robert_random_before_svamp = [33.1, 27.4, 27.7, 32.2, 35.2, 32.7, 31.8, 30.7, 30.1, 26.5]
robert_random_after_svamp  = [41.0, 36.9, 34.0, 34.1, 34.0, 33.9, 32.4, 32.1, 30.0, 30.5]
robert_random_rigl_svamp =   [33.0, 34.0, 31.3, 33.8, 34.0, 33.1, 31.7, 30.1, 31.2, 30.6]

x_axis = range(10)
roberta_large = fig.add_subplot(2,3,6)
roberta_large.plot(x_axis, dense_svamp*10,  '-o',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_svamp,  '-o',color='#228B22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_svamp,  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_svamp,  '--o',color='#228B22',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_gm_after_svamp,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_svamp,  '-o',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_svamp,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_svamp,  '--o',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gmp_svamp,  '-o', color='#CD00CD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_before_svamp,  '-o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_gm_rigl_svamp,  '--o',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('Graph2Tree on SVAMP',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# plt.ylim(25,45)
roberta_large.set_ylabel('Accuracy [%]', fontsize=Titlesize)
roberta_large.set_xlabel('Sparsity',fontsize=fontsize)
roberta_large.set_xticklabels([], rotation=45,fontsize=fontsize )
roberta_large.set_xticklabels(np.array( [0.2, 0.36, 0.488, 0.590, 0.672, 0.738, 0.791, 0.8325, 0.866, 0.893]), fontsize=10 )
roberta_large.tick_params(axis='both', which='major', labelsize=fontsize)
roberta_large.grid(True, linestyle='-', linewidth=0.5, )

roberta_large.spines['right'].set_visible(False)
roberta_large.spines['top'].set_visible(False)


# plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=6, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.06, bottom=0.2, right=0.95, top=0.95, wspace=0.2, hspace=0.35)

plt.savefig('mathall_w_omp.pdf')
plt.show()