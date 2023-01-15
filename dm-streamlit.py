# required download pip package
# pdfkit, wkhtmltopdf

##
# 1. host it at streamlit cloud
# 2. show EDA
# 3. provide a way for user to input data and run predictive model to return results.
# 4. extra -> allow download of a PDF report on EDA.
##

import streamlit as st
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import pickle
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from apyori import apriori


st.set_option('deprecation.showPyplotGlobalUse', False)
import pdfkit

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 500)

st.title("Project Visualize")
markdown_text = '''
Project Members:
1. 1181103230 Loo Chen Zhi
2. 1181103362 Chang See Jie
3. 1181103501 Lim Wei Jie
'''
st.markdown(markdown_text)
st.markdown("---")

st.subheader(f"How we handling the missing values?")
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
st.subheader("Question : Where are the customers located?")
st.subheader("Technique: k-Means Clustering")
st.write("We will perform k-means clustering to group the customers based on their geographic locations (latitude & longitude).")

km_df = merged_df[['latitude', 'longitude']]

k = st.slider("Select the value of k", 1, 10, 5)

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

st.write("Majority center point detection for exact locations")

geolocator = Nominatim(user_agent="geoapiExercises")

for i in range(len(cluster_centers)):
    cluster_center = [cluster_centers[i][0], cluster_centers[i][1]]
    location = geolocator.reverse(cluster_center, exactly_one=True)
    st.write(location.raw['address']['city'] + ' :  ' +str(cluster_centers[i][0]) + ', ' + str(cluster_centers[i][1]))
    
st.markdown("---")

st.subheader("Question: Will a customer purchase drinks in the laundry shop?")

clf_df = merged_df.copy()
st.dataframe(clf_df.head())

# drop unnecessary columns
clf_df.drop(['Date', 'Time', 'latitude', 'longitude'], axis=1, inplace=True)

# one-hot encoding
col_list = [col for col in clf_df.columns.tolist() if clf_df[col].dtype.name == 'object']
df_ob = clf_df[col_list]
clf_df = clf_df.drop(col_list, 1)
df_ob = pd.get_dummies(df_ob)
clf_df = pd.concat([clf_df, df_ob], axis=1)

st.dataframe(clf_df.head())

clf_X = clf_df.drop(['Drinks', 'buyDrinks'], axis=1)
clf_y = clf_df['Drinks']
clf_colnames = clf_X.columns

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

sm = SMOTE()
X_train_smote, y_train_smote = sm.fit_resample(clf_X_train, clf_y_train)

st.write("Original training set shape: ", clf_X_train.shape)
st.write("Resampled training set shape: ", X_train_smote.shape)
st.write("Original test set shape: ", clf_X_test.shape)
st.write("Original target variable shape: ", clf_y_train.shape)
st.write("Resampled target variable shape: ", y_train_smote.shape)

logreg_clf = LogisticRegression(C=c)
logreg_clf.fit(clf_X_train, clf_y_train)
clf_y_pred = logreg_clf.predict(clf_X_test)

def evaluate_model(clf, X_train, y_train, colour, label):
    st.write("Accuracy: {:.3f}".format(accuracy_score(clf_y_test, clf_y_pred)))
    st.write("Accuracy on training set: {:.3f}".format(logreg_clf.score(X_train, y_train)))
    st.write("Accuracy on test set: {:.3f}".format(logreg_clf.score(clf_X_test, clf_y_test)))

    # confusion matrix
    cm = confusion_matrix(clf_y_test, clf_y_pred)
    st.write('**********************')
    st.write('Majority TN =', cm[0][0])
    st.write('Majority FP =', cm[0][1])
    st.write('Majority FN =', cm[1][0])
    st.write('Majority TP =', cm[1][1])
    st.write('**********************')

    # calculate AUC
    prob = clf.predict_proba(clf_X_test)
    prob = prob[:, 1]
    auc = roc_auc_score(clf_y_test, prob)
    st.write('AUC: %.2f' % auc)

    fpr, tpr, thresholds = roc_curve(clf_y_test, prob) # roc curve
    prec, rec, threshold = precision_recall_curve(clf_y_test, prob) # precision-recall curve

    st.write("Precision-Recall: {:.2f}".format(metrics.auc(rec, prec)))

    class_names = ['0', '1']
    disp = plot_confusion_matrix(clf, clf_X_test, clf_y_test, display_labels=class_names, cmap=plt.cm.Blues)
    disp.ax_.set_title("Confusion Matrix")
    st.pyplot()

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

    return fpr, tpr, prec, rec


def logreg_model():
    # Create a slider for the regularization parameter C
    c = st.slider('Regularization parameter C', 0.01, 10.0)

    # train the model
    logreg_clf = LogisticRegression(C=c)
    logreg_clf.fit(clf_X_train, clf_y_train)

    clf_y_pred = logreg_clf.predict(clf_X_test)

    # model accuracy
    st.write('--------------WITHOUT SMOTE--------------')
    logreg_fpr, logreg_tpr, logreg_prec, logreg_rec = evaluate_model(logreg_clf, clf_X_train, clf_y_train, 'orange', 'LogReg')

    # Save the model
    if st.button('Save Model'):
        pickle.dump(logreg_clf, open('logreg_clf.sav', 'wb'))
        st.success('Model saved!')

    # Display coefficients
    coefficients = logreg_clf.coef_
    st.write("Coefficients: ", coefficients)

    # train the model using SMOTE
    logreg_clf = LogisticRegression(C=c)
    logreg_clf.fit(X_train_smote, y_train_smote)
    clf_y_pred = logreg_clf.predict(clf_X_test)

    # model accuracy
    st.write('--------------WITH SMOTE--------------')
    logreg_sm_fpr, logreg_sm_tpr, logreg_sm_prec, logreg_sm_rec = evaluate_model(logreg_clf, X_train_smote, y_train_smote, 'blue', 'LogReg with SMOTE')


if __name__ == '__main__':
    logreg_model()

st.markdown("---")

st.subheader("Question: What is the relationship between the weather conditions and the number of customers at the laundry shop?")

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

X = weather_cus.drop(['count'], axis=1)
y = weather_cus['count']
colnames = X.columns

# select the parameters for the grid search
n_estimators = st.slider("Number of trees in the forest", 10, 100, 50)
max_depth = st.slider("The maximum depth of the tree", 1, 10, 5)
min_samples_leaf = st.slider("The minimum number of samples required to be at a leaf node", 1, 4, 2)

param_grid = {"estimator__n_estimators": [n_estimators],
              "estimator__max_depth": [max_depth],
              "estimator__min_samples_leaf": [min_samples_leaf]}

rf = RandomForestClassifier()
rfe = RFECV(rf, cv=5)
grid_search = GridSearchCV(rfe, param_grid, cv=5)
grid_search.fit(X, y)

# display the results
st.write("Best parameters: {}".format(grid_search.best_params_))
st.write("Best score: {:.2f}".format(grid_search.best_score_))

# # Function to rank the feature importances
# def ranking(ranks, names, order=1):
#     minmax = MinMaxScaler()
#     ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
#     ranks = map(lambda x: round(x,2), ranks)
#     return dict(zip(names, ranks))

# rfe_score = ranking(list(map(float, rfe.support_)), colnames, order=-1)

# rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
# rfe_score = rfe_score.sort_values("Score", ascending = False)

# # Display top 10 features
# st.write('---------Top 10 Features----------')
# st.dataframe(ranking.head(10))



# def generate_report():
#     data = pd.read_csv("dataset.csv")

#     report = ""
#     report += "## Data Summary\n"
#     report += f"Shape of the data: {data.shape}\n"
#     report += f"Columns in the data: {data.columns}\n"
#     report += f"Data Types: \n{data.dtypes}\n"
#     report += f"Missing values: \n{data.isna().sum()}\n"
#     report += "\n## Data Distribution\n"
#     report += "### Numerical Columns\n"
#     report += f"{data.describe()}\n"
#     report += "### Categorical Columns\n"
#     report += f"{data.select_dtypes(include='object').describe()}\n"
#     report += "\n## Correlation\n"
#     report += f"{data.corr()}\n"
#     report += "\n## Distribution of Target Column\n"
#     report += sns.countplot(data["target_column"])
#     report += "\n"
#     return report

# # Generate the EDA report
# report = generate_report()

# # Write the report to the app
# st.write(report)

# # Create a download button
# if st.button('Download Report'):
#     pdfkit.from_string(report, 'report.pdf')
#     st.download('report.pdf')