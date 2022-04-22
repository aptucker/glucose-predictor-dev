
## My Ph.D. Research - Glucose Forecasting with Neural Networks

### Background
Accurate and precise forecasts of glucose for patients with diabetes is an active research area. Many methods have been proposed for forecasting; this repository is focused on neural newtork methods used in my Ph.D. research at the University of Minnesota Earl E. Bakken Medical Devices Center.

My primary research question can be stated as follows: Can a neural network with low computation cost be constructed to accurately forecast glucose from continuous glucose monitor (CGM) data with results that are robust to changes in physiological location? While many forecasting methods have successfully forecasted glucose using glucose data from CGM data used entirely on one arm, there has not been a robust investigation determining whether CGM location between the right and left arms will increase forecast error. To answer this question, two primary NN structures were investigated. 

### Does sensor location affect glucose forecasting error?

First a time-delay feedforward NN was used to produce patient-specific forecasts while isolating CGM location as a testable variable. 5x2 cross validation was used as the comparison method for data gathered from 13 patients with diabetes each wearing a CGM on both arms simultaneously. Results indicated that changes in CGM location can increase forecast errors with algorithms that otherwise perform acceptably. In the figure, every asterisk is an instance where a change in CGM location caused an increase in forecast error. For more, see the publication in the Journal of Diabetes Science and Technology (doi: https://doi.org/10.1177/19322968211018246)

![JDSTErrorSmall](https://media.github.umn.edu/user/20368/files/df934536-d81f-4f6c-a989-4006a0107a05)

![TuckerGRU_Figure_5](https://media.github.umn.edu/user/20368/files/223664f7-d23c-4021-b4f6-cb319e338226)
