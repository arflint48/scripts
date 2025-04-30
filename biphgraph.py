import os, sys, re
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import StrMethodFormatter, FormatStrFormatter, AutoMinorLocator,FixedLocator
from scipy.interpolate import splev, splrep, make_interp_spline, BSpline
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams['savefig.dpi'] = 1000

pes1 = np.loadtxt("6_6_mrci.csv",delimiter=",",dtype="float")
mrcix = pes1[0:,0]
s0 = pes1[0:,2]
s1 = pes1[0:,3]
s2 = pes1[0:,1]
t1 = pes1[6:,4]

xnew = np.linspace(mrcix.min(), mrcix.max(), 400)
xmodnew = np.linspace(mrcix[6:].min(), mrcix[6:].max(), 400)
spls0 = splrep(mrcix,s0)
spls1 = splrep(mrcix,s1)
spls2 = splrep(mrcix,s2)
splt1 = splrep(mrcix[6:],t1)
s0new = splev(xnew, spls0)
s1new = splev(xnew, spls1)
s2new = splev(xnew, spls2)
t1new = splev(xmodnew, splt1)

plt.plot(xnew, s0new, 'k',label=r'1 \textsuperscript{1}A\textsubscript{g}')
plt.plot(xmodnew, t1new, 'g',label=r'1 \textsuperscript{3}B\textsubscript{3g}')
plt.plot(xnew, s1new, 'b',label=r'2 \textsuperscript{1}A\textsubscript{g}')
#plt.plot(xnew, s2new, 'm',label=r'3 \textsuperscript{1}A\textsubscript{g}')

ax = plt.subplot(111)
plt.xlabel(r'C-C intermolecular distance (\r{A})')
plt.ylabel(r'Relative energy (kcal mol\textsuperscript{-1})')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True) 
plt.xticks(fontname='Helvetica')
plt.tick_params(axis='y', which='both', right=False, left=True, labelleft=True) 
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
for pos in ['right', 'top']: 
    plt.gca().spines[pos].set_visible(False) 
ax.set_ylim([-150.0,200.0])
plt.setp(ax,yticks=[-100.0,-50.0,0.0,50.0,100.0,150.0],yticklabels=['-100.0','-50.0','0.0','50.0','100.0','150.0'])
ax.minorticks_on()
plt.yticks([-125.0,-100.0,-75.0,-50.0,-25.0,0.0,25.0,50.0,75.0,100.0,125.0,150.0,175.0],labels=None,minor=True)
plt.xticks([1.75,2.25,2.75,3.25,3.75,4.25,4.75],labels=None,minor=True)
#ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
#ax.tick_params(axis='x', which='minor', bottom=False)
plt.legend(loc=1)

plt.gcf().set_size_inches(12,8)
#plt.show()
plt.savefig('mrci.png',format='png',bbox_inches="tight")

