##
# 1. host it at streamlit cloud
# 2. show EDA
# 3. provide a way for user to input data and run predictive model to return results.
# 4. extra -> allow download of a PDF report on EDA.
##

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 500)

st.set_option('deprecation.showPyplotGlobalUse', False)

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 500)

st.title("Project Visualization")
markdown_text = '''
Project Members:
1. 1181103230 Loo Chen Zhi  01116789079
2. 1181103362 Chang See Jie 0143490382
3. 1181103501 Lim Wei Jie   0125681547
'''
st.markdown(markdown_text)
st.markdown("---")

st.subheader(f"How we handle with the missing values?")
st.markdown("- Fill with 'unknown'")
st.markdown("- Fill with 'median'")
st.markdown("- Fill with 'mode'")
st.markdown("- Fill with '0'")
st.markdown("- Drop all NA for the TotalSpent_RM column")
st.markdown("---")

st.subheader("Data Preprocessing")
st.write("**Initial Dataset**")

# load dataset
df = pd.read_csv("dataset.csv")
weather_df = pd.read_csv("cyberjaya 2015-10-01 to 2016-03-31.csv")
# dataset dataset
# dataset cleaning process
df['Time'] = df['Time'].replace({'15:47;02' : '15:47:02','15:52;08' : '15:52:08'})
df['Date'] =  pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.date
df_obj_col = list(df.select_dtypes(['object']).columns)
df[df_obj_col[2:]] = df[df_obj_col[2:]].apply(lambda x: x.str.strip())
# impute the missing values in object columns with 'unknown' and treat them as a separate category
df[df_obj_col] = df[df_obj_col].fillna('unknown')
# replace the missing values with median
df['Age_Range'] = df['Age_Range'].fillna(df['Age_Range'].median())
df['TimeSpent_minutes'] = df['TimeSpent_minutes'].fillna(df['TimeSpent_minutes'].median())
# replace the missing values with mode
df['Num_of_Baskets'] = df['Num_of_Baskets'].fillna(df['Num_of_Baskets'].mode()[0])
# replace the missing values with 0
df['buyDrinks'] = df['buyDrinks'].fillna(0)
# drop all NA for the TotalSpent_RM column
df.dropna(subset=['TotalSpent_RM'], inplace=True)
df.sort_values(by=['Date', 'Time'], ignore_index=True, inplace=True)
# create a function to perform this data transformation
def addDrinks (row):
    if row['buyDrinks'] > 0:
        return 1
    else:
        return 0

# apply the function to create new column
df['Drinks'] = df.apply(lambda row: addDrinks(row), axis=1)
rows, columns = df.shape
st.write(df.head())
st.write(f"Final dataset consists of **{rows}** rows and **{columns}** columns")

st.write("**Final Dataset with External Data (Weather)**")
# weather dataset
weather_df = pd.read_csv('cyberjaya 2015-10-01 to 2016-03-31.csv')
# dataset cleaning process
weather_df.rename(columns = {'datetime':'Date'}, inplace=True)
weather_df['Date'] =  pd.to_datetime(weather_df['Date'], format='%Y-%m-%d').dt.date
merged_df = df.merge(weather_df, on='Date', how='inner')
merged_df = merged_df.drop(['name', 'sunrise', 'sunset', 'description', 'icon', 'stations', 'snow', 'snowdepth', 'sealevelpressure', 'solarradiation', 'solarenergy', 'uvindex', 'moonphase', 'dew'], axis=1)
merged_df['preciptype'] = merged_df['preciptype'].fillna('None')
merged_df['windgust'] = merged_df['windgust'].fillna(0)
merged_df = merged_df.drop(['severerisk'], axis=1)
rows, columns = merged_df.shape

st.write()
st.write(merged_df.head())
st.write(f"After the pre-processing steps done, the final dataset consists of **{rows}** rows and **{columns}** columns")

st.markdown("---")
st.subheader("Exploratory Data Analysis")

st.write("**Insight 1: The majority of customers are Malay, with a significant number of Indian and Chinese customers as well.**")
st.bar_chart(merged_df["Race"].value_counts())

st.write("**Insight 2: The number of female customers are slighly more than the number of male customers.**")
st.bar_chart(merged_df["Gender"].value_counts())

st.write("**Insight 3: The majority of customers are between 30-40 years old.**")
bins = [10,20,30,40,50,60]
df['Age_Bin'] = pd.cut(df['Age_Range'], bins=bins)
plt.figure(figsize=(12,8))
df['Age_Bin'].value_counts().sort_index().plot(kind='bar')
plt.title('Barchart for Binned Age')
plt.xticks(rotation='horizontal')
plt.yticks(np.arange(0, max(df['Age_Bin'].value_counts())+1, 25))
plt.xlabel('Age')
plt.ylabel('Count')
st.pyplot()

st.markdown("---")



################################################### Clustering ###################################################

st.subheader("Question : Where are the customers located?")
st.caption("To answer this question, K-means clustering is performed to group the customers based on their geographic location, which is the _latitude_ and _longitude_ columns.")

km_df = merged_df[['latitude', 'longitude']]

k = st.slider("Select the value of k", 1, 10, 4)

kmeans = KMeans(n_clusters=k)
kmeans.fit(km_df[['latitude', 'longitude']])

km_df['cluster'] = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

plt.scatter(km_df['latitude'], km_df['longitude'], c=km_df['cluster'])
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='r', s=100)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Clusters of Customers\' Locations')
st.pyplot()

st.caption('‘X’ symbol at the centre of each cluster to represent the centre points.')
st.write("**It can be said that the customers are majorly located in:**")

geolocator = Nominatim(user_agent="geoapiExercises")

for i in range(len(cluster_centers)):
    cluster_center = [cluster_centers[i][0], cluster_centers[i][1]]
    location = geolocator.reverse(cluster_center, exactly_one=True)
    st.write(location.raw['address']['city'] + ' :  ' +str(cluster_centers[i][0]) + ', ' + str(cluster_centers[i][1]))
    
st.markdown("---")



################################################### Classification ###################################################

st.subheader("Question: Will a customer purchase drinks in the laundry shop?")
st.caption('In this question, the _Drinks_ variable will be predicted to know whether or not the customers will buy drinks, where yes=1 and no=0.')
st.caption('Logistic Regression is used to predict the target variable, _Drinks_.')
st.caption('Boruta is used to performing feature selection to identify the most important features from the dataset, the top 10 features are selected')
st.caption('The _Drinks_ variable is imbalanced, so oversampling has to be applied by using SMOTE.')
st.caption('Accuracy, Confusion Matrix, Area Under the Curve (AUC) and Precision-Recall will be used to evaluate the performance.')

clf_df = merged_df.copy()
# st.dataframe(clf_df.head())

# drop unnecessary columns
clf_df.drop(['Date', 'Time', 'latitude', 'longitude'], axis=1, inplace=True)

# one-hot encoding
col_list = [col for col in clf_df.columns.tolist() if clf_df[col].dtype.name == 'object']
df_ob = clf_df[col_list]
clf_df = clf_df.drop(col_list, 1)
df_ob = pd.get_dummies(df_ob)
clf_df = pd.concat([clf_df, df_ob], axis=1)

# st.dataframe(clf_df.head())

clf_X = clf_df.drop(['Drinks', 'buyDrinks'], axis=1)
clf_y = clf_df['Drinks']
clf_colnames = clf_X.columns

##### boruta #####
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced_subsample', max_depth=5)
feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)

feat_selector.fit(clf_X.values, clf_y.values)

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

boruta_score = ranking(list(map(float, feat_selector.ranking_)), clf_colnames, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
boruta_score = boruta_score.sort_values("Score", ascending = False)

boruta_top_10 = boruta_score.Features[:10]

clf_X = clf_df[boruta_top_10]
clf_y = clf_df['Drinks']

clf_X_train, clf_X_test, clf_y_train, clf_y_test = train_test_split(clf_X, clf_y, test_size=0.2, random_state=42)

##### SMOTE #####
sm = SMOTE()
X_train_smote, y_train_smote = sm.fit_resample(clf_X_train, clf_y_train)

# st.write("Original training set shape: ", clf_X_train.shape)
# st.write("Resampled training set shape: ", X_train_smote.shape)
# st.write("Original test set shape: ", clf_X_test.shape)
# st.write("Original target variable shape: ", clf_y_train.shape)
# st.write("Resampled target variable shape: ", y_train_smote.shape)

##### Logistic Regression #####
st.subheader("Logistic Regression")

def evaluate_model(clf, X_train, y_train, colour, label):
    acc = round(accuracy_score(clf_y_test, clf_y_pred), 3)
    acc_train = round(logreg_clf.score(X_train, y_train), 3)
    acc_test = round(logreg_clf.score(clf_X_test, clf_y_test), 3)

    # confusion matrix
    cm = confusion_matrix(clf_y_test, clf_y_pred)
    st.write('Majority TN =', cm[0][0])
    st.write('Majority FP =', cm[0][1])
    st.write('Majority FN =', cm[1][0])
    st.write('Majority TP =', cm[1][1])

    # display coefficients
    coefficients = clf.coef_
    st.write("Coefficients: ", coefficients)

    # calculate AUC
    prob = clf.predict_proba(clf_X_test)
    prob = prob[:, 1]
    auc = roc_auc_score(clf_y_test, prob)

    fpr, tpr, thresholds = roc_curve(clf_y_test, prob) # roc curve
    prec, rec, threshold = precision_recall_curve(clf_y_test, prob) # precision-recall curve

    pr = round(metrics.auc(rec, prec), 2)

    # plot ROC Curve 
    plt.plot(fpr, tpr, color=colour, label=label) 
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    st.pyplot()

    # plot Precision-Recall Curve
    plt.plot(prec, rec, color=colour, label=label) 
    plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot()

    return acc, acc_train, acc_test, cm, auc, pr

# train the model
logreg_clf = LogisticRegression()
logreg_clf.fit(clf_X_train, clf_y_train)
clf_y_pred = logreg_clf.predict(clf_X_test)

st.write("**WITHOUT SMOTE**")

acc, acc_train, acc_test, cm, auc, pr = evaluate_model(logreg_clf, clf_X_train, clf_y_train, 'orange', 'LogReg')

# # Save the model
# if st.button('Save Model'):
#     pickle.dump(logreg_clf, open('logreg_clf.sav', 'wb'))
#     st.success('Model saved!')


# train the model using SMOTE
logreg_sm_clf = LogisticRegression()
logreg_sm_clf.fit(X_train_smote, y_train_smote)
clf_y_pred = logreg_sm_clf.predict(clf_X_test)

st.write("**WITH SMOTE**")

sm_acc, sm_acc_train, sm_acc_test, sm_cm, sm_auc, sm_pr = evaluate_model(logreg_sm_clf, X_train_smote, y_train_smote, 'blue', 'LogReg with SMOTE')

table = {'METRICS': ['Accuracy','Accuracy on training set', 'Accuracy on test set', 'AUC', 'Precision'],
        'WITHOUT SMOTE': [acc, acc_train, acc_test,  auc, pr],
        'WITH SMOTE': [sm_acc, sm_acc_train, sm_acc_test, sm_auc, sm_pr]}

logreg_result = pd.DataFrame(table)
logreg_result.set_index(logreg_result.columns[0], inplace=True)
st.write(logreg_result)

# feature sliders
st.write('')
st.write("**Adjust the features values to predict whether the customer will buy drinks or not:**")

humidity_input = st.slider('humidity', 59.2, 93.1)
winddir_input = st.slider('winddir', 5.1, 357.6)
TimeSpent_minutes_input = st.slider('TimeSpent_minutes', 11.0, 60.0)
Age_Range_input = st.slider('Age_Range', 18.0, 60.0)
cloudcover_input = st.slider('cloudcover', 86.9, 96.5)
visibility_input = st.slider('visibility', 1.2, 10.2)
windspeed_input = st.slider('windspeed', 8.0, 25.9)
temp_input = st.slider('temp', 25.1, 29.9)
feelslikemin_input = st.slider('feelslikemin', 23.0, 29.2)
feelslikemax_input = st.slider('feelslikemax', 33.0, 42.1)

input = [humidity_input, winddir_input, TimeSpent_minutes_input, Age_Range_input, cloudcover_input, 
        visibility_input, windspeed_input, temp_input, feelslikemin_input, feelslikemax_input]

input_pred = logreg_clf.predict([input])
input_sm_pred = logreg_sm_clf.predict([input])

st.write("**Prediction results:**")
st.write('Linear Regression without SMOTE dataset', input_pred[0])
st.write('Linear Regression with SMOTE dataset', input_sm_pred[0])

st.markdown("---")



################################################### Regression ###################################################

st.subheader("Question: What is the relationship between the weather conditions and the number of customers at the laundry shop?")
st.caption('For this question, Linear Regression and Support Vector Regression are used to predict the number of customers based on the weather conditions using the external weather data.') 
st.caption('Recursive Feature Elimination with Cross-Validation (RFECV) is used as a feature selection technique to select the most relevant features for the predictive models. The top 5 features are used for the Linear Regression and Support Vector Regression to do prediction.')
st.caption('Hyperparameter tuning using GridSearch is done on both models to be compared with models without tuning.')
st.caption('The Mean Absolute Error (MAE) and R-Squared (R2) are used to evaluate the models.')

weather_cus = merged_df.drop(['Time', 'Race', 'Gender', 'Body_Size', 'Age_Range', 'With_Kids',
       'Kids_Category', 'Basket_Size', 'Basket_colour', 'Attire',
       'Shirt_Colour', 'shirt_type', 'Pants_Colour', 'pants_type', 'Wash_Item',
       'Washer_No', 'Dryer_No', 'Spectacles', 'TimeSpent_minutes', 'buyDrinks',
       'latitude', 'longitude', 'Num_of_Baskets'], axis=1)

# group by the weather and calcute count of customer on each day
weather_cus = weather_cus.groupby(['Date', 'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
       'feelslike', 'humidity', 'precip', 'precipprob', 'precipcover',
       'preciptype', 'windgust', 'windspeed', 'winddir', 'cloudcover',
       'visibility', 'conditions']).size().reset_index(name='count')

weather_cus.drop(['Date'], axis=1, inplace=True)

# one-hot encoding
col_list = [col for col in weather_cus.columns.tolist() if weather_cus[col].dtype.name == 'object']
df_ob = weather_cus[col_list]
weather_cus = weather_cus.drop(col_list, 1)
df_ob = pd.get_dummies(df_ob)
weather_cus = pd.concat([weather_cus, df_ob], axis=1)
# st.dataframe(weather_cus.head())
# st.write("Data after one-hot encoding")

X = weather_cus.drop(['count'], axis=1)
y = weather_cus['count']
colnames = X.columns

# rf = RandomForestClassifier()
# rfe = RFECV(rf, cv=5)

# param_grid = {"estimator__n_estimators": [10, 50, 100],
#                 "estimator__max_depth": [1, 5, 10],
#                 "estimator__min_samples_leaf": [1, 2, 4]}

# grid_search = GridSearchCV(rfe, param_grid, cv=5)
# grid_search.fit(X, y)

# st.write("Best parameters: {}".format(grid_search.best_params_))
# st.write("Best score: {:.2f}".format(grid_search.best_score_))

ht_rf = RandomForestClassifier(n_estimators=10, max_depth=1, min_samples_leaf=1)
rf.fit(X, y)
rfe = RFECV(ht_rf, min_features_to_select=1, cv=2)

rfe.fit(X, y)

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)

rfe_top_5 = rfe_score.Features[:5]

reg_X = weather_cus[rfe_top_5]
reg_y = weather_cus['count']

reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(reg_X, reg_y, test_size=0.3, random_state=50)

st.write("**Linear Regression**")

linreg = LinearRegression()
linreg.fit(reg_X_train, reg_y_train)
reg_y_pred = linreg.predict(reg_X_test)

# calculate the mean absolute error and r-squared of the predictions
mae_lin = mean_absolute_error(reg_y_test, reg_y_pred)
r2_lin = r2_score(reg_y_test, reg_y_pred)

# hyperparameter tuning
linreg = LinearRegression()
parameters = {'fit_intercept':[True,False]} # define the hyperparameters and their possible values
grid_search = GridSearchCV(linreg, parameters, cv=5) # perform grid search with cross-validation
grid_search.fit(reg_X_train, reg_y_train)
linreg = LinearRegression(**grid_search.best_params_) # train the final model on the entire training set
linreg.fit(reg_X_train, reg_y_train)
reg_y_pred = linreg.predict(reg_X_test) # evaluate the final model on the test set

# calculate the mean absolute error and r-squared of the predictions
mae_lin_ht = mean_absolute_error(reg_y_test, reg_y_pred)
r2_lin_ht = r2_score(reg_y_test, reg_y_pred)

table = {'Metrics': ['Mean Absolute Error','R2 Score'],
        'Before Hyperparameter tuning': [mae_lin, r2_lin],
        'After Hyperparameter tuning': [mae_lin_ht, r2_lin_ht]}

lr_result = pd.DataFrame(table)
lr_result.set_index(lr_result.columns[0], inplace=True)
st.write(lr_result)

st.write("**Support Vector Regression**")

# create the SVR model
svr = SVR(kernel='linear', C=1.0, epsilon=0.1)

# fit the model to the training data
svr.fit(reg_X_train, reg_y_train)

# use the model to make predictions on the test data
reg_y_pred = svr.predict(reg_X_test)

# calculate the mean absolute error and r-squared of the predictions
mae_svr = mean_absolute_error(reg_y_test, reg_y_pred)
r2_svr = r2_score(reg_y_test, reg_y_pred)

svr = SVR()

param_grid = {"C": [0.1, 1, 10], 
              "kernel": ["linear", "poly", "rbf"],
              "degree":[1, 2, 3],
              "epsilon": [0.01, 0.1, 1]}
    
svr_grid_search = GridSearchCV(svr, param_grid, cv=5)
svr_grid_search.fit(reg_X_train, reg_y_train)

# create the SVR model
svr = SVR(**svr_grid_search.best_params_)

# fit the model to the training data
svr.fit(reg_X_train, reg_y_train)

# use the model to make predictions on the test data
reg_y_pred = svr.predict(reg_X_test)

# calculate the mean absolute error and r-squared of the predictions
mae_svr_ht = mean_absolute_error(reg_y_test, reg_y_pred)
r2_svr_ht = r2_score(reg_y_test, reg_y_pred)

table = {'Metrics': ['Mean Absolute Error','R2 Score'],
        'Before Hyperparameter tuning': [mae_svr, r2_svr],
        'After Hyperparameter tuning': [mae_svr_ht, r2_svr_ht]}

svr_result = pd.DataFrame(table)
svr_result.set_index(svr_result.columns[0], inplace=True)
st.write(svr_result)

st.write('')
st.write('To answer the question, both regression models are poorly performing. Both models are not suitable for predicting the number of customers based on the weather conditions. Therefore, we can conclude that the relationship between weather and the number of customers is weak.')
