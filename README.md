# Apply Random Forest to Classify House Loan Workflow  

## 1. Background and Target
Financial organizations need to decide if it is risky to loan money to someone.  
They can use supervized machine learning models like Random Forest(RF) to do this classification task.  
Input data: features, like people's demographics(gender, age..), band credits, education etc.  
Output data: classification classes(Y, N)  

## 2. Exploratory Data Analysis (EDA)
Understand the data through different angle
### 2.1 Number of features & feature types
<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/0_1_feature_viz.png" width="35%" height="35%"/>
</div>
### 2.2 Specific feature's statistics
<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/0_2_feature_viz.png" width="35%" height="35%"/>
</div>
### 2.3 Number of nulls in the data
<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/0_3_feature_viz.png" width="35%" height="35%"/>
</div>

<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/tree_visulization.png" width="90%" height="100%"/>
</div>

## 1. Box plot with matplotlib.pyplot to identify outliers
continuous features

<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/tree_visulization.png" width="90%" height="100%"/>
</div>

## 2. Distribution plots with seaborn is distplot & pairplot 
continuous & categorical features

<div align="left">
<img src="https://github.com/GuilinXie/DataVisualization/blob/master/image/3_1_tip_distribution.png" width="35%" height="35%"/>
<img src="https://github.com/GuilinXie/DataVisualization/blob/master/image/3_2_pairplot_for_smoker.png" width="35%" height="35%">
</div>


reference:  
1 https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset/tasks  
2 https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf  
3 https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c  
4 https://blog.csdn.net/lumugua/article/details/83450005
