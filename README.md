
## My Ph.D. Research - Glucose Forecasting with Neural Networks

Accurate and precise forecasts of glucose for patients with diabetes is an active research area. Many methods have been proposed for forecasting; this repository is focused on neural newtork methods used in my Ph.D. research at the University of Minnesota Earl E. Bakken Medical Devices Center.

My primary research question can be stated as follows: Can a neural network with low computation cost be constructed to accurately forecast glucose from continuous glucose monitor (CGM) data with results that are robust to changes in physiological location? While many forecasting methods have successfully forecasted glucose using glucose data from CGM data used entirely on one arm, there has not been a robust investigation determining whether CGM location between the right and left arms will increase forecast error. To answer this question, two primary NN structures were investigated. 

First a time-delay feedforward NN was used to produce patient-specific forecasts while isolating CGM location as a testable variable. 5x2 cross validation was used as the comparison method for data gathered from 13 patients with diabetes each wearing a CGM on both arms simultaneously. Results indicated that changes in CGM location can increase forecast errors with algorithms that otherwise perform acceptably. 


![TuckerGRU_Figure_5](https://media.github.umn.edu/user/20368/files/223664f7-d23c-4021-b4f6-cb319e338226)
