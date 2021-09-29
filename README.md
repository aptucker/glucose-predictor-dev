## Error Reducing Glucose Predictor Dev

#### Current Status <br>
Models Constructed
1. JDST Model - Alpaydin forward prop. equation with 1 hidden layer; same construction as JDST analysis but in TF
2. Parallel Model - JDST structure in parallel with left and right arms each taking one "tower"
3. Circadian Model 1 - JDST structure with 3 additional inputs lagged 24 hr
4. Seqential with 2 hidden layers - Alpaydin forward prop. equation with 2 hidden layers (note use more epochs, first takes additional time to converge)

General Notes 
1. Each individual patient has a file which pulls from the customModels and customLayers modules
2. TF models cannot be pickled, they are created in each individual patient file and not saved with the patient object