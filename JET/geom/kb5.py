
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../bib/')
import bib_geom 

# -----------------------------------------------------------------------------

R,Z = bib_geom.get_vessel_JET()

plt.figure()
plt.plot(R, Z, 'b')
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.axes().set_aspect('equal')

# -----------------------------------------------------------------------------

Ri,_,Zi,_ = bib_geom.get_los_JET(inside = False)
Ri2,Rf,Zi2,Zf = bib_geom.get_los_JET(inside = True)

for (ri,rf,zi,zf,ri2,zi2) in zip(Ri,Rf,Zi,Zf,Ri2,Zi2):
    plt.plot((ri,ri2), (zi,zi2), 'g', linewidth=.7)
    plt.plot((ri2,rf), (zi2,zf), 'r', linewidth=.7)

plt.text(3.5,2.2,'KB5V')
plt.text(4.5,-.3,'KB5H')
plt.tight_layout()
plt.savefig('kb5.png',dpi=300,bbox_inches='tight')
plt.show()
