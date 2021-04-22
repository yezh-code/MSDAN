# MSDAN
A  code of MSDAN

# Paper
Multi-source Domain Adaption for Health Degradation Monitoring of Lithium-ion Batteries

# Usage
main.py: It is feasible to run and can save the training loss and prediction results for plotting. <br />
plot.py: It is used to visualize prediction results that authors presented in the manuscript.<br />
data_process.py: It is used to load data and load the batch data in the training process.<br />
function.py: It is used to define the reverse layer for adversarial learning.<br />
MMD.py: It is used to define the MMD loss between source domain and target domain.<br />
Model.py: It is used to define the base model (i.e., MSDAN).<br />

# Procedure
(1) Run the main.py to train MSDAN and save the training loss and the prediction results. <br />
(2) Run the plot.py to visualize prediction results and the training loss.<br />
(3) The prediction results are saved in result folder, the trained model is saved in model folder, the visualization results are saved in figure folder.<br />

# Data
The details about the source and target domain for each experiment are listed in the following Table.<br />
No.	Transfer direction	Source domain	Target domain<br />
1	C2&C3→C1	Condition 2 and Condition 3	Condition 1<br />
2	C1&C3→C2	Condition 1 and Condition 3	Condition 2<br />
3	C1&C2→C3	Condition 1 and Condition 2	Condition 3<br />
  For the target domain, three batteries are used for training, and the rest one battery is used for testing. It should be pointed out that this code is an example of transfer direction C2&C3→C1, and the battery #5 is used as the testing battery, other three batteries (i.e., #6, #7, #18) in target domain are used for training.<br />
  In the data folder, there are three domains (i.e., C1.mat, C2.mat, C3.mat), which consists of all batteries in the corresponding domain. C1.mat consists of the four batteries (#5, #6, #7, #18) in condition 1. Moreover, C1_tr1.mat denotes that three batteries (i.e., #6, #7, #18) are used for training when C1 are used as target domain, and C1_te1.mat denotes the rest battery, i.e., #5.  For example, in the transfer direction C2&C3→C1, we aim to monitor the health condition of #5. In this case, C2.mat and C3.mat are used as the source domain for training, and C1_tr1.mat is used as target domain for training, C1_te1.mat is used as target domain for testing. The detailed description about the data is summarized as following:<br />
Data name	Definition	  Data name	 Definition	     Data name    Definition<br />
C1.mat	        #5, #6, #7, #18	  C2.mat	 #29, #30, #31, #32  C3.mat	  #45, #46, #47, #48<br />
C1._tr1.mat	#6, #7, #18	  C2._tr1.mat	 #30, #31, #32	     C3._tr1.mat  #46, #47, #48<br />
C1._te1.mat	#5	          C2._te1.mat	 #29,	             C3._te1.mat  #45<br />
C1._tr2.mat	#5, #7, #18	  C2._tr2.mat	 #29, #31, #32	     C3._tr2.mat  #45, #47, #48<br />
C1._te2.mat	#6	          C2._te2.mat	 #30	             C3._te2.mat  #46<br />
C1._tr3.mat	#5, #6, #18	  C2._tr3.mat	 #29, #30, #32	     C3._tr3.mat  #45, #46, #48<br />
C1._te3.mat	#7	          C2._te3.mat	 #31	             C3._te3.mat  #47,<br />
C1._tr4.mat	#5, #6, #7	  C2._tr4.mat	 #29, #30, #31,	     C3._tr4.mat  #45, #46, #47<br />
C1._te4.mat	#18	          C2._te4.mat	 #32	             C3._te4.mat  #48<br />

# Environment
torch 1.2.0, Python 3.6, Numpy 1.16.2<br />
