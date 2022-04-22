
## My Ph.D. Research - Glucose Forecasting with Neural Networks

### Background
Accurate and precise forecasts of glucose for patients with diabetes is an active research area. Many methods have been proposed for forecasting; this repository is focused on neural newtork (NN) methods used in my Ph.D. research at the University of Minnesota Earl E. Bakken Medical Devices Center.

My primary research question can be stated as follows: Can a neural network with low computation cost be constructed to accurately forecast glucose from continuous glucose monitor (CGM) data with results that are robust to changes in physiological location? While many forecasting methods have successfully forecasted glucose using glucose data from CGM data used entirely on one arm, there has not been a robust investigation determining whether CGM location between the right and left arms will increase forecast error. To answer this question, two primary NN structures were investigated. 

### Research Aim 1: Does sensor location affect glucose forecasting error?

First, a time-delay feedforward NN was used to produce patient-specific forecasts while isolating CGM location as a testable variable. 5x2 cross validation was used as the comparison method for data gathered from 13 patients with diabetes each wearing a CGM on both arms simultaneously. Results indicated that changes in CGM location can increase forecast errors with algorithms that otherwise perform acceptably. 

Consider comparisons of algorithms trained and tested on different arms: left arm trained - left arm tested versus right arm trained - left arm tested and vice versa. In the figure, every asterisk is an instance where a change in CGM location caused an increase in forecast error within one of the comparison groups (left-left vs right-left and right-right vs left-right). These results held independent of overall glucose variance, for more, see the publication in the Journal of Diabetes Science and Technology (doi: https://doi.org/10.1177/19322968211018246)

![JDSTErrorSmall](https://media.github.umn.edu/user/20368/files/df934536-d81f-4f6c-a989-4006a0107a05)

### Research Aim 2: Mitigating the effects of sensor location

With the novel observation that CGM location can increase glucose forecasting error, my next research aim consisted of designing a neural network which would be robust to switches in CGM position. Using gated recurrent units (GRUs - Cho, et al. 2014), a new NN was able to accurately forecast glucose while being agnostic to CGM location changes. Results indicated no statistically significant increases in error due to changes with the GRU NN. These results are currently under review at the Journal of Diabetes Science and Technology.

![TuckerGRU_Figure_5](https://media.github.umn.edu/user/20368/files/223664f7-d23c-4021-b4f6-cb319e338226)

### Research Aim 3: Reducing computational cost for the GRU NN

Improving forecasting robustness came with a distinct tradeoff in computational cost resulting in increased training time. Using a control systems engineering technique known as linear quadratic regulated control, I was able to reduce training time for the GRU glucose forecasting algorithm by 10 times over the standard method. With this method, training time was also reduced over both a GRU NN with an Adam optimizer as well as the original time-delay feedforward NN. In the figure, the GRUNN w/LQR LR curve is the result of the new method. 

![timeTrialN71400AllMethods](https://media.github.umn.edu/user/20368/files/1eb58e82-2a20-4df2-b52f-f52ae2fcb306)


