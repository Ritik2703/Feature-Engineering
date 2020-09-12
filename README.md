# Feature-Engineering
# Hare Krishna

# Encoding - https://www.geeksforgeeks.org/feature-encoding-techniques-machine-learning/
Process of coverting data from one form to another required form.
# Type of Encoding -
## 1. Nominal Encoding  (Unordered Groups)
   1. One Hot Encoding
   2. One Hot Encoding with many Categorical variables
   3. Mean Encoding
## 2. Ordinal Encoding  (Ordered Groups)
   1. Label Encoding
   2. Target guided Ordinal Encoding
   
   
### One Hot Encoding -
   Spliting of categories to different columns.
   Put ‘0 for others and ‘1’ as an indicator for the appropriate column.

#### Using sklearn

```
from sklearn.preprocessing import OneHotEncoder 
enc = OneHotEncoder() 
# tranforming the column after fitting 
enc = enc.fit_transform(df[['nom_0']]).toarray() 
# converting arrays to a dataframe 
encoded_colm = pd.DataFrame(enc) 
# concating dataframes  
df = pd.concat([df, encoded_colm], axis = 1)  
# removing the encoded column. 
df = df.drop(['nom_0'], axis = 1)  
df.head(10) 
```

#### using pandas

```
df = pd.get_dummies(df, prefix = ['nom_0'], columns = ['nom_0']) 
df.head(10) 
```
  
### One Hot Encoding with many Categorical variables
In this we select the 10 topmost most repeated categories.

USed by a team in kdd competition and win the competition with good accuracy

### Frequency encoding
encode considering the frequency distribution

### Mean encoding
Target encoding is good because it picks up values that can explain the target. It is used by most kagglers in their competitions. The basic idea to replace a categorical value with the mean of the target variable.
```
df.insert(5, "Target", [0, 1, 1, 0, 0, 1, 0, 0, 0, 1], True)  
# importing TargetEncoder 
from category_encoders import TargetEncoder 
Targetenc = TargetEncoder() 
# tranforming the column after fitting 
values = Targetenc.fit_transform(X = df.nom_0, y = df.Target) 
# concating values with dataframe 
df = pd.concat([df, values], axis = 1) 
df.head(10) 
```


### Label encoding
Label encoding algorithm is quite simple and it considers an order for encoding.
```
from sklearn.preprocessing import LabelEncoder   
le = LabelEncoder() 
df['ord_2'] = le.fit_transform(df['ord_2']) 
sns.set(style ="darkgrid") 
sns.countplot(df['ord_2']) 
```
### Target Guided Ordinal encoding

If we have more ordinal data, then we find first mean then give the rank based on ot.

# Scaling/Normalization -  https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/

Needed when in dataset we had multiple features spanning varying degrees of magnitude, range, and units.
So it used to convert values to lower range.

## Normalization
Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1. It is also known as Min-Max scaling.

```
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min

>>> from sklearn.preprocessing import MinMaxScaler
>>> scaler = MinMaxScaler()
```

# Standardization
It is another scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation.
If mean is 0 and standard deviation is 1, then it follow standard normal distribution.

```
z = (x - u) / s

>>> from sklearn.preprocessing import StandardScaler
>>> scaler = StandardScaler()
```

# Handling Missing Values- https://analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/
1. Delete rows.
2. Replace with Mean, Median and Mode
3. Apply Classsifier Algorithms
4. Apply Unsupervise Algorithms







