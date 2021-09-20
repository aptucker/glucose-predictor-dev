# Written and designed by Aaron Tucker 
#
#
#
# ----------------------------------------------------------------------------
"""
Patient creation and pickling
"""

import sys
sys.path.append('../')

import patient as pat
import pickle


L1  =  pat.createPatient('..\\..\\raw_data\\CGM1Left.csv')
L3  =  pat.createPatient('..\\..\\raw_data\\CGM3Left.csv')
L6  =  pat.createPatient('..\\..\\raw_data\\CGM6Left.csv')
L10 = pat.createPatient('..\\..\\raw_data\\CGM10Left.csv')
L11 = pat.createPatient('..\\..\\raw_data\\CGM11Left.csv')
L14 = pat.createPatient('..\\..\\raw_data\\CGM14Left.csv')
L18 = pat.createPatient('..\\..\\raw_data\\CGM18Left.csv')
L21 = pat.createPatient('..\\..\\raw_data\\CGM21Left.csv')
L28 = pat.createPatient('..\\..\\raw_data\\CGM28Left.csv')
L30 = pat.createPatient('..\\..\\raw_data\\CGM30Left.csv')
L33 = pat.createPatient('..\\..\\raw_data\\CGM33Left.csv')
L36 = pat.createPatient('..\\..\\raw_data\\CGM36Left.csv')
L38 = pat.createPatient('..\\..\\raw_data\\CGM38Left.csv')

R1  =  pat.createPatient('..\\..\\raw_data\\CGM1Right.csv')
R3  =  pat.createPatient('..\\..\\raw_data\\CGM3Right.csv')
R6  =  pat.createPatient('..\\..\\raw_data\\CGM6Right.csv')
R10 = pat.createPatient('..\\..\\raw_data\\CGM10Right.csv')
R11 = pat.createPatient('..\\..\\raw_data\\CGM11Right.csv')
R14 = pat.createPatient('..\\..\\raw_data\\CGM14Right.csv')
R18 = pat.createPatient('..\\..\\raw_data\\CGM18Right.csv')
R21 = pat.createPatient('..\\..\\raw_data\\CGM21Right.csv')
R28 = pat.createPatient('..\\..\\raw_data\\CGM28Right.csv')
R30 = pat.createPatient('..\\..\\raw_data\\CGM30Right.csv')
R33 = pat.createPatient('..\\..\\raw_data\\CGM33Right.csv')
R36 = pat.createPatient('..\\..\\raw_data\\CGM36Right.csv')
R38 = pat.createPatient('..\\..\\raw_data\\CGM38Right.csv')

with open("patient1.pickle", "wb") as f:
    pickle.dump([L1, R1], f)
    
with open("patient2.pickle", "wb") as f:
    pickle.dump([L3, R3], f)
    
with open("patient3.pickle", "wb") as f:
    pickle.dump([L6, R6], f)
    
with open("patient4.pickle", "wb") as f:
    pickle.dump([L10, R10], f)
    
with open("patient5.pickle", "wb") as f:
    pickle.dump([L11, R11], f)
    
with open("patient6.pickle", "wb") as f:
    pickle.dump([L14, R14], f)

with open("patient7.pickle", "wb") as f:
    pickle.dump([L18, R18], f)
    
with open("patient8.pickle", "wb") as f:
    pickle.dump([L21, R21], f)
    
with open("patient9.pickle", "wb") as f:
    pickle.dump([L28, R28], f)
    
with open("patient10.pickle", "wb") as f:
    pickle.dump([L30, R30], f)
    
with open("patient11.pickle", "wb") as f:
    pickle.dump([L33, R33], f)
    
with open("patient12.pickle", "wb") as f:
    pickle.dump([L36, R36], f)
    
with open("patient13.pickle", "wb") as f:
    pickle.dump([L38, R38], f)
    
