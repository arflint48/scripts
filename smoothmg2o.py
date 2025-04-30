import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import StrMethodFormatter, FormatStrFormatter
from scipy.interpolate import splev, splrep, make_interp_spline, BSpline

plt.rcParams["font.family"] = "sans-serif"

#step_sizes=np.array([0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100])
#nu1=np.array([656.0,879.4,892.0,894.7,896.1,897.4,898.7,900.2,901.9,903.7,905.7,908.0,910.4,913.0,915.8,918.8,922.0,925.4,929.0,932.8])
#nu2=np.array([475.5,478.3,478.5,478.8,479.2,479.7,480.2,480.8,481.5,482.3,483.2,484.1,485.2,486.3,487.5,488.9,490.3,491.8,493.3,495.0])
#nu3=np.array([184.1,96.2,91.9,91.3,91.4,91.7,92.0,92.5,93.0,93.6,94.3,95.0,95.7,96.6,97.4,98.4,99.3,100.4,101.4,102.5])
step_sizes=np.array([0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100])
nu1=np.array([879.4,892.0,894.7,896.1,897.4,898.7,900.2,901.9,903.7,905.7,908.0,910.4,913.0,915.8,918.8,922.0,925.4,929.0,932.8])
nu2=np.array([478.3,478.5,478.8,479.2,479.7,480.2,480.8,481.5,482.3,483.2,484.1,485.2,486.3,487.5,488.9,490.3,491.8,493.3,495.0])
nu3=np.array([96.2,91.9,91.3,91.4,91.7,92.0,92.5,93.0,93.6,94.3,95.0,95.7,96.6,97.4,98.4,99.3,100.4,101.4,102.5])
ssnew=np.linspace(step_sizes.min(), step_sizes.max(), 50)
spl1=splrep(step_sizes,nu1)
nu1sm=splev(ssnew,spl1)
spl2=splrep(step_sizes,nu2)
nu2sm=splev(ssnew,spl2)
spl3=splrep(step_sizes,nu3)
nu3sm=splev(ssnew,spl3)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, sharey=False)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.ylabel("Frequency (cm$^{-1}$)",labelpad=20.0)
plt.xlabel(r'Step Size ($\AA$)')
plt.subplots_adjust(hspace=0.17)

ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
ax1.set_xticklabels([])
ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
ax2.set_xticklabels([])
ax3.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))

ax3.xaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(3.0))

ax1.plot(ssnew,nu1sm,'#0F52BA',zorder=1)
ax1.scatter(step_sizes,nu1,c='r',zorder=2)
ax2.plot(ssnew,nu2sm,'#0F52BA',zorder=1)
ax2.scatter(step_sizes,nu2,c='r',zorder=2)
ax3.plot(ssnew,nu3sm,'#0F52BA',zorder=1)
ax3.scatter(step_sizes,nu3,c='r',zorder=2)

t1 = chr(957) + "$_1$"
t2 = chr(957) + "$_2$"
t3 = chr(957) + "$_3$"

ax1.set_title(t1)
ax2.set_title(t2)
ax3.set_title(t3)

plt.gcf().set_size_inches(12,6)
plt.gcf().set_dpi(300)
#plt.show()
plt.savefig('mgfunds.png',format='png',bbox_inches="tight")
