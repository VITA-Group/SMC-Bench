import matplotlib.pyplot as plt
import numpy as np
import os, re
from matplotlib.pyplot import figure
fig = figure(num=None, figsize=(19, 6), dpi=150, facecolor='w', edgecolor='k')
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

dense_mawps = [88.23]

robert_gmp_mawps = [87.08, 86.46, 87.29, 87.14, 86.06, 84.84, 84.79, 84.64, 84.48, 84.06]
robert_omp_after_mawps = [88.59, 88.39, 88.59, 87.55, 86.41, 85.42, 86.41, 86.09, 84.32, 81.25]
robert_omp_before_mawps = [87.81, 86.46, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08]
robert_omp_rigl_mawps  = [87.81, 86.09,  2.08, 2.08, 2.08, 2.08, 2.08, 2.08, 2.08,]

robert_random_after_mawps  = [87.23, 87.14, 86.72, 86.01, 85.73, 84.58, 83.85, 83.39, 82.66, 81.35]
robert_random_before_mawps = [81.93, 82.66, 82.60, 82.24, 83.02, 83.33, 81.35, 82.31, 80.26, 80.98]
robert_random_rigl_mawps   = [83.33, 83.96, 83.90, 84.58, 84.21, 82.91, 81.14, 81.45, 80.57, 81.45]

robert_snip_mawps = [85.83, 85.47, 85.31, 84.37, 84.11,  84.73, 83.59, 83.64, 82.5, 80.20]

robert_snip_rigl_mawps = [85.47, 85.67, 85.15, 85.26, 83.59, 84.79, 84.16, 83.43, 82.95, 81.95]

robert_lth_mawps = [87.86, 88.54, 89.47, 88.85, 87.60, 86.14, 84.68, 84.01, 82.86, 79.53]



roberta_large = fig.add_subplot(1,3,1)
roberta_large.plot(x_axis, dense_mawps*10,  '-o',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_mawps,  '-o',color='#00FF00',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_mawps,  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_mawps,  '--o',color='#00FF00',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_omp_after_mawps,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_mawps,  '-o',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_mawps,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_mawps,  '--o',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gmp_mawps,  '-o', color='magenta',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_omp_before_mawps,  '-o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_omp_rigl_mawps,  '--o',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('GTS on MAWPS',fontsize=Titlesize)
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




roberta_large = fig.add_subplot(1,3,2)
roberta_large.plot(x_axis, dense_asdiv*10,  '-o',   label='Dense model',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_asdiv,  '-o',   label='SNIP (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_asdiv,  '-o',   label='LTH (After)',color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_asdiv,  '--o',   label='SNIP+RIGL (Before)',color='#00FF00',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_gm_after_asdiv,  '-o',   label='One-Shot LRR (After)',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_asdiv,  '-o',   label='Random (Before)',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_asdiv,  '-o',   label='Random LRR (After)',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_asdiv,  '--o',   label='Random+RIGL (Before)',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gmp_asdiv_seed2,  '-o',   label='GMP (During)',color='magenta',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gm_before_asdiv,  '-o',   label='OMP (Before)' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gm_rigl_asdiv,  '--o',   label='OMP+RIGL (Before)',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('GTS on ASDiv-A',fontsize=Titlesize)
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
roberta_large = fig.add_subplot(1,3,3)
roberta_large.plot(x_axis, dense_svamp*10,  '-o',color='black',linewidth=linewidth, markersize=markersize, )

roberta_large.plot(x_axis, robert_snip_svamp,  '-o',color='#00FF00',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_lth_svamp,  '-o', color='orange',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_snip_rigl_svamp,  '--o',color='#00FF00',linewidth=linewidth, markersize=markersize )
roberta_large.plot(x_axis, robert_gm_after_svamp,  '-o',color='#00BFFF',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_before_svamp,  '-o',color='brown',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_after_svamp,  '-o',color='#0072BD',linewidth=linewidth, markersize=markersize, )
roberta_large.plot(x_axis, robert_random_rigl_svamp,  '--o',color='brown',linewidth=linewidth, markersize=markersize)
roberta_large.plot(x_axis, robert_gmp_svamp,  '-o', color='magenta',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gm_before_svamp,  '-o' ,color='#bcbd22',linewidth=linewidth, markersize=markersize, )
# roberta_large.plot(x_axis, robert_gm_rigl_svamp,  '--o',color='#bcbd22',linewidth=linewidth, markersize=markersize )


roberta_large.set_title('GTS on SVAMP',fontsize=Titlesize)
roberta_large.set_xticks(range(10))
# roberta_large.axes.set_ylabel().set_visible(True)
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


# plt.tight_layout()
fig.legend(loc='lower center', bbox_to_anchor=(0.0, 0.0, 1, 1), fancybox=False, shadow=False, ncol=6, fontsize=fontsize, frameon=False)
fig.subplots_adjust(left=0.05 , bottom=0.3, right=0.99, top=0.95, wspace=0.2, hspace=0.2)

plt.savefig('graph2tree_no_omp_before.pdf')
plt.show()