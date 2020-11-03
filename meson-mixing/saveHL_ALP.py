import numpy as np
import sys
import itertools
import multiprocessing

rdet = 2.5 #m -- radius of the detector 
DetDist = 574.0 #m -- distance between the meson source (target) and the detector
ThetaDet = rdet/DetDist

MMesTEST = float(sys.argv[1]) #Mass of the parent meson (\pi^0, \eta, \eta^\prime) in GeV: Can be found in the "ParentMesons" directory
MpName = str(sys.argv[2]) #Location of the Pythia-output (converted to .npy) of the parent meson four-vectors
MpVec = np.load(MpName)
    
def Boost(bx, by, bz, gam, b):
    return [[gam, gam*bx, gam*by, gam*bz], [gam*bx, 1.+(gam-1.)*bx**2/b**2, (gam-1.)/b**2 * bx*by, (gam-1.)/b**2 * bx*bz], [gam*by, (gam-1.)/b**2*bx*by, 1.+(gam-1.)/b**2*by**2, (gam-1.)/b**2*by*bz], [gam*bz, (gam-1.)/b**2*bx*bz, (gam-1.)/b**2*by*bz, 1.+(gam-1.)/b**2*bz**2]]


#----------------------------------------------------------------------------------------------
# Generate a flux of Axions from a Parent Meson Distribution Mixing with the Axions
# Save the energy/direction of the accepted Axions -- cut on direction later
#----------------------------------------------------------------------------------------------
def AlpFromMes(mALP, meslist, mmes):
    retvec = []
    
    for j in range(len(meslist)):
        if np.mod(j, 100000) == 0:
            print(mALP,j)

        pmag = np.sqrt(meslist[j][1]**2 + meslist[j][2]**2 + meslist[j][3]**2) #Three-momentum magnitude of the parent meson, which the axion inherits
        Epi0init = np.sqrt(pmag**2 + mmes**2) #Check the energy of the parent meson -- if it is smaller than the mass of the axion we are considering, it gets zero energy/large angle, which will be rejected
        if Epi0init < mALP:
            retvec.append([0.0, 6.0])
        else:
            en = np.sqrt(pmag**2 + mALP**2) #Lab-frame energy of the axion
            thetaALP = np.arctan(np.sqrt(meslist[j][1]**2 + meslist[j][2]**2)/np.abs(meslist[j][3])) #Lab-frame angle of the outgoing Axion

            retvec.append([en, thetaALP])
        
    return retvec

NPOT = 10.0 * 1.47e21 #10 years times 1.47e21 POT per Year
cMP = float(sys.argv[3]) #Number (per POT) of the parent meson produced
ThetaPrefac = float(sys.argv[4]) #Prefactor that enters the Axion/Meson mixing-squared

def ThetaAMesSq(ma, mmes, gammes):
    return ThetaPrefac*ma**4/((ma**2-mmes**2)**2 + mmes**2*gammes**2)

#------------------------------------------------------------------------------------------------------------------------------------------------
# Function that returns the flux (number passing through the detector) of Axions at the Near Detector as a histogram with respect to Axion Energy
# Only axions passing through the near detector are accepted
# Returned object 'hl' is a one-dimensional array, with the number passing through the ND with energy between E_i and E_{i+1} -- 1 GeV spacing
#------------------------------------------------------------------------------------------------------------------------------------------------
def NAFn(theta):
    MaT = theta[0]
    aVecs = AlpFromMes(MaT, MpVec, MMesTEST)
        
    aVecsAcc = []
    for k in range(len(aVecs)):
        if aVecs[k][1] < ThetaDet:
            aVecsAcc.append(aVecs[k][0])
    NALP0 = NPOT*cMP*ThetaAMesSq(MaT, MMesTEST, 1.0e-8*MMesTEST)*(float(len(aVecsAcc)))/(float(len(aVecs))) #Calculate the total normalization, accounting for acceptance
    hl = (NALP0/(float(len(aVecsAcc))))*np.histogram(aVecsAcc, bins=120, range=[0.0, 120.0])[0]

    print(MaT, NALP0)
    return hl

#Scan over m_a between 10 MeV and 10 GeV, 400 points
lmamin = -2.0
lmamax = 1.0
nma = 400
maTab = [10**(lmamin + (lmamax-lmamin)/nma*j) for j in range(nma+1)]

#Parallelize the calculation of the histograms
paramlist = list(itertools.product(maTab))
pool = multiprocessing.Pool(12)
res = pool.map(NAFn, paramlist)

#Save the output -- FNOut is just a filename that the two "_hl" and "_masses" get appended to
FNOut = str(sys.argv[5])
HLSave = FNOut + "_hl"
MSave = FNOut + "_masses"

np.save(HLSave, res)
np.save(MSave, maTab)