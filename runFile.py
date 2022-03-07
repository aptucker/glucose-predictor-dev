# Spyder Run File
#
#
#
# ----------------------------------------------------------------------------

# cellList = ['Imports and Data Loading',
#             'Load w/Previous Results',
#             'GRU H=1 Model',
#             'Save Results']

cellList = ['Imports and Data Loading',
            'Load w/Previous Results',
            'Parallel with Circadian Inputs',
            'Save Results']

# cellList = ['Imports and Data Loading',
#             'Load w/o Previous Results',
#             'JDST Model Definition',
#             'GRU H=1 Model',
#             'Save Results']

# cellList = ['Imports and Data Loading',
#             'Load w/Previous Results',
#             'JDST Model Definition',
#             'Save Results']

fileList = ['C:/Code/glucose-predictor-dev/patient1.py',
            'C:/Code/glucose-predictor-dev/patient2.py',
            'C:/Code/glucose-predictor-dev/patient3.py',
            'C:/Code/glucose-predictor-dev/patient4.py',
            'C:/Code/glucose-predictor-dev/patient5.py',
            'C:/Code/glucose-predictor-dev/patient6.py',
            'C:/Code/glucose-predictor-dev/patient7.py',
            'C:/Code/glucose-predictor-dev/patient8.py',
            'C:/Code/glucose-predictor-dev/patient9.py',
            'C:/Code/glucose-predictor-dev/patient10.py',
            'C:/Code/glucose-predictor-dev/patient11.py',
            'C:/Code/glucose-predictor-dev/patient12.py',
            'C:/Code/glucose-predictor-dev/patient13.py']

for i in range(len(fileList)):
    for e in range(len(cellList)):
        runcell(cellList[e], fileList[i])
    



