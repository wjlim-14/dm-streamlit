# drop columns that are not related to weather
weather_cus = merged_df.drop(['Time', 'Race', 'Gender', 'Body_Size', 'Age_Range', 'With_Kids',
       'Kids_Category', 'Basket_Size', 'Basket_colour', 'Attire',
       'Shirt_Colour', 'shirt_type', 'Pants_Colour', 'pants_type', 'Wash_Item',
       'Washer_No', 'Dryer_No', 'Spectacles', 'TimeSpent_minutes', 'buyDrinks',
       'latitude', 'longitude', 'Num_of_Baskets', 
       'Drinks'], axis=1)

# group by the weather and calcute count of customer on each day
weather_cus = weather_cus.groupby(['Date', 'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
       'feelslike', 'humidity', 'precip', 'precipprob', 'precipcover',
       'preciptype', 'windgust', 'windspeed', 'winddir', 'cloudcover',
       'visibility', 'conditions']).size().reset_index(name='count')

weather_cus.drop(['Date'], axis=1, inplace=True)
weather_cus.head(5)

# one-hot encoding
col_list = [col for col in weather_cus.columns.tolist() if weather_cus[col].dtype.name == 'object']
df_ob = weather_cus[col_list]
weather_cus = weather_cus.drop(col_list, 1)
df_ob = pd.get_dummies(df_ob)
weather_cus = pd.concat([weather_cus, df_ob], axis=1)
weather_cus.head()

X = weather_cus.drop(['count'], axis=1)
y = weather_cus['count']
colnames = X.columns

rf = RandomForestClassifier()
rfe = RFECV(rf, cv=5)

param_grid = {"estimator__n_estimators": [10, 50, 100],
              "estimator__max_depth": [1, 5, 10],
              "estimator__min_samples_leaf": [1, 2, 4]}

grid_search = GridSearchCV(rfe, param_grid, cv=5)
grid_search.fit(X, y)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best score: {:.2f}".format(grid_search.best_score_))

ht_rf = RandomForestClassifier(n_estimators=10, max_depth=1, min_samples_leaf=1)
# ht_rf = RandomForestClassifier(n_estimators=100, max_depth=1, min_samples_leaf=2)
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

print('---------Top 10----------')
display(rfe_score.head(10))

print('---------Bottom 10----------')
rfe_score.tail(10)

reg_X = weather_cus[rfe_top_5]
reg_y = weather_cus['count']

# split the data into training and test sets
reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(reg_X, reg_y, test_size=0.3, random_state=50)

# Create the linear regression model
linreg = LinearRegression()

# Define the hyperparameters and their possible values
parameters = {'fit_intercept':[True,False], 'normalize':[True,False]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(linreg, parameters, cv=5)
grid_search.fit(reg_X_train, reg_y_train)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

svr = SVR()

param_grid = {"C": [0.1, 1, 10], 
              "kernel": ["linear", "poly", "rbf"],
              "degree":[1, 2, 3],
              "epsilon": [0.01, 0.1, 1]}
    
svr_grid_search = GridSearchCV(svr, param_grid, cv=5)
svr_grid_search.fit(reg_X_train, reg_y_train)
print("Best parameters: {}".format(svr_grid_search.best_params_))