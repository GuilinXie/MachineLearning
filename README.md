# Apply Random Forest to Classify House Loan Workflow  

## 1. Background and Target
* Financial organizations need to decide if it is risky to loan money to someone.  
* They can use supervized machine learning models like Random Forest(RF) to do this classification task.  
* Input data: features, like people's demographics(gender, age..), band credits, education etc.  
* Output data: classification classes(Y, N)  

## 2. Exploratory Data Analysis (EDA)
* Understand the data through different angles
### 2.1 number of features & features' type
<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/0_1_feature_viz.png" width="20%" height="20%"/>
</div>    

### 2.2 feature's statistics
<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/0_2_feature_viz.png" width="25%" height="25%"/>
</div>    

### 2.3 number of nulls in the data
<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/0_3_feature_viz.png" width="20%" height="20%"/>
</div>  

### 2.4 features visualization   
#### features distribution
<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/1_feature_dist_viz.png" width="35%" height="35%"/>
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/2_feature_viz.png" width="35%" height="35%"/>
</div>  

#### features & labels correlation
```
sns.set(style="ticks", color_codes=True)
sns.pairplot(train[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Status']], hue="Loan_Status", palette="coolwarm")
plt.show()
```

<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/3_feature_label_pairplot_pic.png" width="50%" height="50%"/>
</div> 
  
    
## 2. Data Preprocessing
* Clean, filter, fill(imputer), transform, normalization, encoder(one-hot, LabelEncoder), feature-selection(selectKbest,chi-squared)
   
## 3. Model training & tuning params   
* RF model configuration example
```
rf = RandomForestClassifier(
                       bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=4, max_features=4,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=20,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
```
* The parameters for RF to tune is usually **max_depth**, **max_features**, and **n_estimators**
   
## 4. Evaluate model performance

### 4.1 Possible Metrics
| Metrics        | Classification           | Regression  |
| ------------- |:-------------:| -----:|
|   1    | Precison | MSE |
| 2      | Recall      |   RMSE |
|3 | F1     |    MAE |
| 4 | Accuracy      |    R^2 |
| 5 | ROC/AUC     |     |
| 6| Confusion Matrix    |     |

* Note:  
**ROC/AUC** for binary classify  
Sensitivity(Recall, True Positive Rate)->ROC y-axis   
1-Specificity(False Positive Rate) -> ROC x-axis  

### 4.2 Clasification Report    

```  
from sklearn.metrics import classification_report

result = classification_report(y_test, y_pred, target_names=rf[1].classes_)
for line in result.split("\r\n"):
    print(line)
```  

<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/classification_report.png" width="50%" height="50%"/>
</div>  

### 4.3 confusion matrix  
* In this project, I used confusion matrix to check the model performance.  
```  
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

matrix = confusion_matrix(y_test, y_pred)
ax = sns.heatmap(matrix,annot=True,cbar=True,center=100.0,xticklabels=rf[1].classes_.tolist(),yticklabels=rf[1].classes_.tolist())
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom, top)
plt.title("Confusion Matrix")
```  
<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/confusion%20matrix.png" width="50%" height="50%"/>
</div>

## 5. Interpret Models  
* LR model is not easy to interpret because of the random selection of samples and features,     
* However, we could still pick some of the individul trees to visulize the prediction logics.     
* In this project, I chose one tree(max_depth=4, max_features=4) from the forest to check how features contribute to the final classification.  

<div align="left">
<img src="https://github.com/GuilinXie/MachineLearning/blob/master/image/tree_visulization.png" width="90%" height="80%"/>
</div>  
  
```  
estimator = rf[1].estimators_[1]
from sklearn.tree import export_graphviz

export_graphviz(estimator, 
                out_file='tree.dot', 
                feature_names = features,
                class_names = rf[1].classes_,
                rounded = True, proportion = False, 
                precision = 2, filled = True)  
```  
dot -Tpdf tree2.dot -o tree4.pdf -Gdpi=600 # use this command in cmd win10 to convert dot to pdf    
dot -Tpng tree2.dot -o tree5.png -Gdpi=600 # use this command in cmd win10 to convert dot to png    
  
## 6. reference:  
1 https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset/tasks  
2 https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf  
3 https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c  
4 https://blog.csdn.net/lumugua/article/details/83450005
