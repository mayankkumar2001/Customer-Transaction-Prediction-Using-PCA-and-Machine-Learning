# Customer Transaction Prediction Using PCA and Machine Learning

## Project Overview
This project builds a binary classification workflow to predict the target variable from customer transaction data. It focuses on data exploration, feature preparation, dimensionality reduction, class balancing, and classifier comparison.

## Objectives
- Explore transaction data and feature distributions.
- Prepare numerical features for modelling.
- Apply Principal Component Analysis (PCA) for dimensionality reduction.
- Address class imbalance using SMOTE.
- Train and evaluate classification models.

## Dataset
The notebook uses `train.csv`, with `target` as the prediction target and `ID_code` as an identifier removed during feature selection. The dataset contains numerical variables such as `var_0`, `var_1`, and others.

## Project Workflow
1. Inspect data quality, feature distributions, correlations, and duplicates.
2. Scale numerical variables and remove the identifier column.
3. Apply PCA to reduce feature dimensionality.
4. Use SMOTE to address target-class imbalance.
5. Train, tune, and evaluate classification models.

## Machine Learning Models
- Logistic Regression
- Logistic Regression with Bagging
- Multi-Layer Perceptron (MLP)
- XGBoost

## Evaluation
The notebook uses accuracy, precision, recall, F1-score, and classification reports. Reported MLP results in the notebook are approximately 93.87% training accuracy and 90.47% testing accuracy; results can vary with execution and environment.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

## Project Notebook
Open the project notebook: `Transaction(1).ipynb`.

## Complete Project Code
The complete source code from the notebook is included below. Expand the section to view it.

<details>
<summary>View complete project code</summary>

### Code Cell 1

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
```


### Code Cell 2

```python
data = pd.read_csv("train.csv")
```


### Code Cell 3

```python
data.head()
```


### Code Cell 4

```python
data.describe()
```


### Code Cell 5

```python
data.info()
```


### Code Cell 6

```python
data.tail()
```


### Code Cell 7

```python
data.target.value_counts()
```


### Code Cell 8

```python
dist = data.iloc[:,2:102]
plt.figure(figsize=(20,25),facecolor='white') # Define canvas size
plotno = 1 # Counter
for column in dist: # Itreting the feature
    if plotno<=100: # Condition
        plt.subplot(10,10,plotno) # plotting 100 boxplot with 10 rows and 10 columN
        sns.histplot(dist[column],kde=True,color='orange')
        plt.xlabel(column,fontsize=15) # assigning name to x-axis and font size is 20
    plotno+=1
plt.tight_layout()
plt.show()
```


### Code Cell 9

```python
dist2 = data.iloc[:,103:]
plt.figure(figsize=(20,25),facecolor='white')
plotno = 1
for col in dist2:
    if plotno<=100:
        plt.subplot(10,10,plotno)
        sns.histplot(dist2[col],kde=True,color='orange')
        plt.xlabel(col,fontsize=15)
    plotno = plotno+1
plt.tight_layout()
plt.show()
```


### Code Cell 10

```python
plt.figure(figsize=(16,6))
sns.set_style('whitegrid')
features = data.columns.values[1:202]
plt.title('Distribution of mean values per row in the data')
plt.xlabel('mean value')
plt.ylabel('pdf value')
sns.distplot(data[features].mean(axis=1),color='ORANGE', kde=True,bins=120)
plt.show()
```


### Code Cell 11

```python
plt.figure(figsize=(16,6))
sns.set_style('whitegrid')
features = data.columns.values[1:202]
plt.title('Distribution of Standard Deviation per row in the data')
plt.xlabel('mean value')
plt.ylabel('pdf value')
sns.distplot(data[features].std(axis=1),color='ORANGE', kde=True,bins=120)
plt.show()
```


### Code Cell 12

```python
### Data preprocessing
```


### Code Cell 13

```python
data.isnull().sum()
```


### Code Cell 14

```python
co = data.iloc[:0:203]
```


### Code Cell 15

```python
def missing_value(data):
    null_value = data.isnull().sum()
    percentage = (data.isnull().sum()/data.isnull().count()*100)
    null_value = pd.concat([null_value,percentage],axis=1,keys=['Missing Number','Missing Percentage'])
    return null_value
pd.set_option('display.max_rows',None)
missing_value(data)
```


### Code Cell 16

```python
###3.FEATURE SCALING
```


### Code Cell 17

```python
from sklearn.preprocessing import RobustScaler # Import library
scaler = RobustScaler() # Object creation
data[['var_0','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8','var_9','var_10','var_11','var_12','var_13',
      'var_14','var_15','var_16','var_17','var_18','var_19','var_20','var_21','var_22','var_23','var_24','var_25','var_26',
      'var_27','var_28','var_29','var_30','var_31','var_32','var_33','var_34','var_35','var_36','var_37','var_38','var_39',
      'var_40','var_41','var_42','var_43','var_44','var_45','var_46','var_47','var_48','var_49','var_50','var_51','var_52',
      'var_53','var_54','var_55','var_56','var_57','var_58','var_59','var_60','var_61','var_62','var_63','var_64','var_65',
      'var_66','var_67','var_68','var_69','var_70','var_71','var_72','var_73','var_74','var_75','var_76','var_77','var_78',
      'var_79','var_80','var_81','var_82','var_83','var_84','var_85','var_86','var_87','var_88','var_89','var_90','var_91',
      'var_92','var_93','var_94','var_95','var_96','var_97','var_98','var_99','var_100','var_101','var_102','var_103',
      'var_104','var_105','var_106','var_107','var_108','var_109','var_110','var_111','var_112','var_113','var_114',
      'var_115','var_116','var_117','var_118','var_119','var_120','var_121','var_122','var_123','var_124','var_125',
      'var_126','var_127','var_128','var_129','var_130','var_131','var_132','var_133','var_134','var_135','var_136','var_137',
      'var_138','var_139','var_140','var_141','var_142','var_143','var_144','var_145','var_146','var_147','var_148','var_149',
      'var_150','var_151','var_152','var_153','var_154','var_155','var_156','var_157','var_158','var_159','var_160','var_161',
      'var_162','var_163','var_164','var_165','var_166','var_167','var_168','var_169','var_170','var_171','var_172','var_173',
      'var_174','var_175','var_176','var_177','var_178','var_179','var_180','var_181','var_182','var_183','var_184','var_185',
      'var_186','var_187','var_188','var_189','var_190','var_191','var_192','var_193','var_194','var_195','var_196','var_197',
      'var_198','var_199',]] = scaler.fit_transform(data[['var_0','var_1','var_2','var_3','var_4','var_5','var_6','var_7',
      'var_8','var_9','var_10','var_11','var_12','var_13',
      'var_14','var_15','var_16','var_17','var_18','var_19','var_20','var_21','var_22','var_23','var_24','var_25','var_26',
      'var_27','var_28','var_29','var_30','var_31','var_32','var_33','var_34','var_35','var_36','var_37','var_38','var_39',
      'var_40','var_41','var_42','var_43','var_44','var_45','var_46','var_47','var_48','var_49','var_50','var_51','var_52',
      'var_53','var_54','var_55','var_56','var_57','var_58','var_59','var_60','var_61','var_62','var_63','var_64','var_65',
      'var_66','var_67','var_68','var_69','var_70','var_71','var_72','var_73','var_74','var_75','var_76','var_77','var_78',
      'var_79','var_80','var_81','var_82','var_83','var_84','var_85','var_86','var_87','var_88','var_89','var_90','var_91',
      'var_92','var_93','var_94','var_95','var_96','var_97','var_98','var_99','var_100','var_101','var_102','var_103',
      'var_104','var_105','var_106','var_107','var_108','var_109','var_110','var_111','var_112','var_113','var_114',
      'var_115','var_116','var_117','var_118','var_119','var_120','var_121','var_122','var_123','var_124','var_125',
      'var_126','var_127','var_128','var_129','var_130','var_131','var_132','var_133','var_134','var_135','var_136','var_137',
      'var_138','var_139','var_140','var_141','var_142','var_143','var_144','var_145','var_146','var_147','var_148','var_149',
      'var_150','var_151','var_152','var_153','var_154','var_155','var_156','var_157','var_158','var_159','var_160','var_161',
      'var_162','var_163','var_164','var_165','var_166','var_167','var_168','var_169','var_170','var_171','var_172','var_173',
      'var_174','var_175','var_176','var_177','var_178','var_179','var_180','var_181','var_182','var_183','var_184','var_185',
      'var_186','var_187','var_188','var_189','var_190','var_191','var_192','var_193','var_194','var_195','var_196','var_197',
      'var_198','var_199']])
```


### Code Cell 18

```python
data.head()
```


### Code Cell 19

```python
data.drop('ID_code',axis=1,inplace=True)
```


### Code Cell 20

```python
data.head()
```


### Code Cell 21

```python
data.corr()
```


### Code Cell 22

```python
plt.figure(figsize=(40,45))
sns.heatmap(data.corr(),annot=True,cmap='cool')
plt.show()
```


### Code Cell 23

```python
# To get highly corelated feature 
pd.set_option('display.max_rows',None)
corrmat = data.corr()
corrmat = corrmat.abs().unstack()
corrmat = corrmat.sort_values(ascending=False)
corrmat = corrmat [corrmat >= 0.9] # Set threshold 0.9
corrmat = corrmat [corrmat < 1]
corrmat = pd.DataFrame(corrmat).reset_index()
corrmat.columns = ['feature1','feature2','corr']
corrmat
```


### Code Cell 24

```python
#Their is no highly correlated feature in dataset
```


### Code Cell 25

```python
####CHECKING DUPLICATES
```


### Code Cell 26

```python
data.duplicated().sum()
```


### Code Cell 27

```python
###Their is no duplicates avialable on data
```


### Code Cell 28

```python
# Getting the optimal value of pca
from sklearn.decomposition import PCA
pca = PCA() # Object creation
principlecomponent = pca.fit_transform(data)  # Fitting the data
plt.figure(figsize=(10,7))
sns.set_style('darkgrid')
plt.plot(np.cumsum(pca.explained_variance_ratio_),color ='k')
plt.xlabel('Number of components',fontsize=20)
plt.ylabel('Variance(%)',fontsize=20) # for each componant
plt.title('Explained Variance',fontsize=20)
plt.show()
```


### Code Cell 29

```python
# Selecting the components
pca = PCA(n_components=175)
new_data = pca.fit_transform(data)
new_data
```


### Code Cell 30

```python
# Create a new dataframe
principle_df = pd.DataFrame(data=new_data,columns=['pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8','pca9','pca10',
                            'pca11','pca12','pca13','pca14','pca15','pca16','pca17','pca18','pca19','pca20','pca21','pca22',
                            'pca23','pca24','pca25','pca26','pca27','pca28','pca29','pca30','pca31','pca32','pca33','pca34',
                            'pca35','pca36','pca37','pca38','pca39','pca40','pca41','pca42','pca43','pca44','pca45','pca46',
                            'pca47','pca48','pca49','pca50','pca51','pca52','pca53','pca54','pca55','pca56','pca57','pca58',
                            'pca59','pca60','pca61','pca62','pca63','pca64','pca65','pca66','pca67','pca68','pca69','pca70',
                            'pca71','pca72','pca73','pca74','pca75','pca76','pca77','pca78','pca79','pca80','pca81','pca82',
                            'pca83','pca84','pca85','pca86','pca87','pca88','pca89','pca90','pca91','pca92','pca93','pca94',
                            'pca95','pca96','pca97','pca98','pca99','pca100','pca101','pca102','pca103','pca104','pca105',
                            'pca106','pca107','pca108','pca109','pca110','pca111','pca112','pca113','pca114','pca115','pca116',
                            'pca117','pca118','pca119','pca120','pca121','pca122','pca123','pca124','pca125','pca126','pca127',
                            'pca128','pca129','pca130','pca131','pca132','pca133','pca134','pca135','pca136','pca137','pca138',
                            'pca139','pca140','pca141','pca142','pca143','pca144','pca145','pca146','pca147','pca148','pca149',
                            'pca150','pca151','pca152','pca153','pca154','pca155','pca156','pca157','pca158','pca159','pca160',
                            'pca161','pca162','pca163','pca164','pca165','pca166','pca167','pca168','pca169','pca170',
                            'pca171','pca172','pca173','pca174','pca175'])
principle_df.head()
```


### Code Cell 31

```python
#DEFINE INDEPENDANT AND DEPENDANT VERIABLE
```


### Code Cell 32

```python
X = principle_df
y = data.target
```


### Code Cell 33

```python
X.head()
```


### Code Cell 34

```python
from collections import Counter
from imblearn.over_sampling import SMOTE #SMOTE(synthetic minority oversampling techinque)
sm = SMOTE() # obeject creation
print("unbalanced data   :  ",Counter(y))
X_sm,y_sm = sm.fit_resample(X,y)
print("balanced data:    :",Counter(y_sm))
```


### Code Cell 35

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_sm,y_sm,random_state=42,test_size=0.30) 
# 30% data given to testing because the large amount of data
```


### Code Cell 36

```python
X_train.shape
```


### Code Cell 37

```python
X_test.shape
```


### Code Cell 38

```python
y_train.shape
```


### Code Cell 39

```python
y_test.shape
```


### Code Cell 40

```python
from sklearn.linear_model import LogisticRegression
```


### Code Cell 41

```python
log_reg = LogisticRegression()
```


### Code Cell 42

```python
log_reg.fit(X_train,y_train)
```


### Code Cell 43

```python
log_test_predict = log_reg.predict(X_test)
```


### Code Cell 44

```python
log_train_predict = log_reg.predict(X_train)
```


### Code Cell 45

```python
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,classification_report,confusion_matrix
log_train_accuracy = accuracy_score(log_train_predict,y_train)
print("Training accuracy of Logistic regression model",log_train_accuracy*100)
print("Logistic regression training Classification report: \n",classification_report(log_train_predict,y_train))
```


### Code Cell 46

```python
log_test_accuracy = accuracy_score(log_test_predict,y_test)
print("Testing accuracy of Logistic regression model",log_test_accuracy*100)
print("Logistic regression testing Classification report: \n",classification_report(log_test_predict,y_test))
```


### Code Cell 47

```python
# In heaert customer transaction prediction f1 score is more important
log_f1_score = recall_score(log_test_predict,y_test)
log_f1_score*100
```


### Code Cell 48

```python
from sklearn.ensemble import BaggingClassifier

# Step 1: Create Logistic Regression object
log_reg1 = LogisticRegression(max_iter=1000)

# Step 2: Create Bagging classifier
bagg = BaggingClassifier(
    estimator=log_reg1,
    n_estimators=50,
    random_state=42
)

# Step 3: Fit training data
bagg.fit(X_train, y_train)

# Step 4: Predict on test data
bagg_predict = bagg.predict(X_test)
```


### Code Cell 49

```python
log_bagg_accuracy = accuracy_score(bagg_predict,y_test)
print("Testing accuracy of Logistic regession model after bagging",log_bagg_accuracy*100)
print("Logistic regression testing Classification report: \n",classification_report(bagg_predict,y_test))
```


### Code Cell 50

```python

bagg_f1_score = f1_score(bagg_predict,y_test)
bagg_f1_score*100
```


### Code Cell 51

```python
# Step:1 Importing library and object creation
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(60,3),
                      learning_rate='constant',
                      max_iter=100,
                      random_state=42)
```


### Code Cell 52

```python
# Step:2 Fitting the training data
model.fit(X_train,y_train)
```


### Code Cell 53

```python
# Step:3 Predicting the probability
mlp_prdict_probability = model.predict_proba(X_test)
mlp_prdict_probability
```


### Code Cell 54

```python
# Step:4 Prediction on test data
mlp_test_predict = model.predict(X_test)

# Step:5 Prediction on training data
mlp_train_predict = model.predict(X_train)
```


### Code Cell 55

```python
# Step:1 importing library and object creation
import xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier()

# Step:2 fitting the training data
xgb.fit(X_train,y_train)
```


### Code Cell 56

```python
# Step:3 Prediction on training data
xgb_train_predict = xgb.predict(X_train)

# Step:4 Prediction on testing data
xgb_test_predict = xgb.predict(X_test)
```


### Code Cell 57

```python
xgb_train_accuracy = accuracy_score(xgb_train_predict,y_train)
print("Training accuracy XGB classifier:",xgb_train_accuracy*100)
print("Classification report of training:"'\n',classification_report(xgb_train_predict,y_train))
```


### Code Cell 58

```python
xgb_test_accuracy = accuracy_score(xgb_test_predict,y_test)
print("Testing accuracy XGB classifier:",xgb_test_accuracy*100)
print("Classification report of training:"'\n',classification_report(xgb_test_predict,y_test))
```


### Code Cell 59

```python
xgb_f1_score = f1_score(xgb_test_predict,y_test)
print("F1 score of xgb:",xgb_f1_score*100)
```


</details>

## Resume Project Title
**Customer Transaction Prediction Using PCA and Machine Learning**

**Resume description:** Developed a transaction classification pipeline using Robust Scaling, PCA-based dimensionality reduction, SMOTE for class balancing, and Logistic Regression, MLP, and XGBoost models; evaluated performance using F1-score and classification metrics.
