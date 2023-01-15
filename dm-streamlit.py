# required download pip package
# pdfkit, wkhtmltopdf

##
# 1. host it at streamlit cloud
# 2. show EDA
# 3. provide a way for user to input data and run predictive model to return results.
# 4. extra -> allow download of a PDF report on EDA.
##
!pip install seaborn

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from geopy.geocoders import Nominatim
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
rows, columns = df.shape
st.write(df.head())
st.write(f"Final dataset consists of **{rows}** rows and **{columns}** columns")

st.write("**External Data - Weather**")
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
st.bar_chart(merged_df["Race"].value_counts())

st.bar_chart(merged_df["Gender"].value_counts())

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
st.subheader("Clustering Analysis")
st.write("Question 1: Where are the customers located?")

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

geolocator = Nominatim(user_agent="geoapiExercises")

for i in range(len(cluster_centers)):
    cluster_center = [cluster_centers[i][0], cluster_centers[i][1]]
    location = geolocator.reverse(cluster_center, exactly_one=True)
    st.write(str(cluster_centers[i][0]) + ', ' + str(cluster_centers[i][1]) + ': ' +location.raw['address']['city'])


st.markdown("---")


st.title('Weather vs Customer Count')

# Create a function to drop unnecessary columns
def drop_cols(df):
    return df.drop(['Time', 'Race', 'Gender', 'Body_Size', 'Age_Range', 'With_Kids',
                   'Kids_Category', 'Basket_Size', 'Basket_colour', 'Attire',
                   'Shirt_Colour', 'shirt_type', 'Pants_Colour', 'pants_type', 'Wash_Item',
                   'Washer_No', 'Dryer_No', 'Spectacles', 'TimeSpent_minutes', 'buyDrinks',
                   'latitude', 'longitude', 'Num_of_Baskets', 
                   'Drinks'], axis=1)

# Create a function to group and count customers by weather
def group_weather(df):
    weather_cus = df.groupby(['Date', 'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
                              'feelslike', 'humidity', 'precip', 'precipprob', 'precipcover',
                              'preciptype', 'windgust', 'windspeed', 'winddir', 'cloudcover',
                              'visibility', 'conditions']).size().reset_index(name='count')
    weather_cus.drop(['Date'], axis=1, inplace=True)
    return weather_cus

# Use Streamlit to create a user interface to upload the data
# st.set_page_config(page_title="Weather Customer", page_icon=":guardsman:", layout="wide")

# Create a function to select object columns
def select_obj_cols(df):
    col_list = [col for col in df.columns.tolist() if df[col].dtype.name == 'object']
    df_ob = df[col_list]
    df = df.drop(col_list, 1)
    df_ob = pd.get_dummies(df_ob)
    df = pd.concat([df, df_ob], axis=1)
    return df

# Create a function to perform random forest classification
def random_forest_classification(X, y):
    rf = RandomForestClassifier()
    rfe = RFECV(rf, cv=5)
    param_grid = {"estimator__n_estimators": [10, 50, 100],
                  "estimator__max_depth": [1, 5, 10],
                  "estimator__min_samples_leaf": [1, 2, 4]}
    grid_search = GridSearchCV(rfe, param_grid, cv=5)
    grid_search.fit(X, y)
    st.write("Best parameters: {}".format(grid_search.best_params_))
    st.write("Best score: {:.2f}".format(grid_search.best_score_))
    ht_rf = RandomForestClassifier(n_estimators=10, max_depth=1, min_samples_leaf=1)
    rf.fit(X, y)
    rfe = RFECV(ht_rf, min_features_to_select=1, cv=2)
    rfe.fit(X, y)

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

group_weather = select_obj_cols(group_weather)
X = group_weather.drop(['count'], axis=1)
y = group_weather['count']
colnames = X.columns
rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)
rfe_top_5 = rfe_score.Features[:5]
st.subheader("RFE Top 5 Features:")
st.dataframe(rfe_top_5)
reg_X = group_weather[rfe_top_5]
reg_y = group_weather['count']
# split the data into training and test sets
reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(reg_X, reg_y, test_size=0.3, random_state=50)
st.subheader("Linear Regression Model:")
linreg = LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False]}
grid_search = GridSearchCV(linreg, parameters, cv=5)
grid_search.fit(reg_X_train, reg_y_train)
st.write("Best parameters: ", grid_search.best_params_)
st.write("Best score: ", grid_search.best_score_)
st.subheader("SVR Model:")
svr = SVR()
param_grid = {"C": [0.1, 1, 10], 
                "kernel": ["linear", "poly", "rbf"],
                "degree":[1, 2, 3],
                "epsilon": [0.01, 0.1, 1]}
svr_grid_search = GridSearchCV(svr, param_grid, cv=5)
svr_grid_search.fit(reg_X_train, reg_y_train)
st.write("Best parameters: {}".format(svr_grid_search.best_params_))

# Make predictions on the test data
reg_y_pred = grid_search.predict(reg_X_test)
svr_y_pred = svr_grid_search.predict(reg_X_test)

# Calculate the evaluation metrics for linear regression
linreg_mae = mean_absolute_error(reg_y_test, reg_y_pred)
linreg_mse = mean_squared_error(reg_y_test, reg_y_pred)
linreg_r2 = r2_score(reg_y_test, reg_y_pred)

# Calculate the evaluation metrics for SVR
svr_mae = mean_absolute_error(reg_y_test, svr_y_pred)
svr_mse = mean_squared_error(reg_y_test, svr_y_pred)
svr_r2 = r2_score(reg_y_test, svr_y_pred)

# Display the evaluation metrics
st.write("Linear Regression Evaluation Metrics:")
st.write("Mean Absolute Error: ", linreg_mae)
st.write("Mean Squared Error: ", linreg_mse)
st.write("R-Squared: ", linreg_r2)
st.write("SVR Evaluation Metrics:")
st.write("Mean Absolute Error: ", svr_mae)
st.write("Mean Squared Error: ", svr_mse)
st.write("R-Squared: ", svr_r2)

# st.set_page_config(page_title="Weather Customer", page_icon=":guardsman:", layout="wide")
# st.subheader("Upload your data")

# file_upload = st.file_uploader("Upload a CSV file", type=["csv"])
# if file_upload is not None:
#     weather_cus = pd.read_csv(file_upload)
#     weather_cus = select_obj_cols(weather_cus)
#     X = weather_cus.drop(['count'], axis=1)
#     y = weather_cus['count']
#     st.subheader("Random Forest Classification:")
#     random_forest_classification(X, y)

# need configure API
# st.subheader("Upload your data")

# file_upload = st.file_uploader("Upload a CSV file", type=["csv"])
# if file_upload is not None:
#     merged_df = pd.read_csv(file_upload)
#     weather_cus = drop_cols(merged_df)
#     weather_cus = group_weather(weather_cus)

#     # Show the result
#     st.subheader("Results:")
#     st.dataframe(weather_cus.head(5))


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
