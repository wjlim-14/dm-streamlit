clf_df = merged_df.copy()

# drop unnecessary columns
clf_df.drop(['Date', 'Time', 'latitude', 'longitude'], axis=1, inplace=True)

# one-hot encoding
col_list = [col for col in clf_df.columns.tolist() if clf_df[col].dtype.name == 'object']
df_ob = clf_df[col_list]
clf_df = clf_df.drop(col_list, 1)
df_ob = pd.get_dummies(df_ob)
clf_df = pd.concat([clf_df, df_ob], axis=1)

clf_df.head()



# boruta feature selection
clf_X = clf_df.drop(['Drinks', 'buyDrinks'], axis=1)
clf_y = clf_df['Drinks']
clf_colnames = clf_X.columns

# prepare boruta classifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced_subsample', max_depth=5)
feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)

# fir boruta classifier to data
feat_selector.fit(clf_X.values, clf_y.values)


# get the ranking of the features returned by Boruta
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

boruta_score = ranking(list(map(float, feat_selector.ranking_)), clf_colnames, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
boruta_score = boruta_score.sort_values("Score", ascending = False)

# extract top 10 features
boruta_top_10 = boruta_score.Features[:10]


# split the dataset into X (feature variables) and y (target variable)
clf_X = clf_df[boruta_top_10]
clf_y = clf_df['Drinks']

# split the data into training and test sets (80% for training and 20% for testing)
clf_X_train, clf_X_test, clf_y_train, clf_y_test = train_test_split(clf_X, clf_y, test_size=0.2, random_state=42)

# apply SMOTE to the training set
sm = SMOTE()
X_train_smote, y_train_smote = sm.fit_resample(clf_X_train, clf_y_train)

print(clf_X_train.shape)
print(clf_y_train.shape)
print(clf_X_test.shape)
print(clf_y_test.shape)


def evaluate_model(clf, X_train, y_train, colour, label):
    print("Accuracy: {:.3f}".format(accuracy_score(clf_y_test, clf_y_pred)))
    print("Accuracy on training set: {:.3f}".format(logreg_clf.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(logreg_clf.score(clf_X_test, clf_y_test)))

    # confusion matrix
    cm = confusion_matrix(clf_y_test, clf_y_pred)
    print('**********************')
    print('Majority TN =', cm[0][0])
    print('Majority FP =', cm[0][1])
    print('Majority FN =', cm[1][0])
    print('Majority TP =', cm[1][1])
    print('**********************')

    # calculate AUC
    prob = clf.predict_proba(clf_X_test)
    prob = prob[:, 1]
    auc = roc_auc_score(clf_y_test, prob)
    print('AUC: %.2f' % auc)

    fpr, tpr, thresholds = roc_curve(clf_y_test, prob) # roc curve
    prec, rec, threshold = precision_recall_curve(clf_y_test, prob) # precision-recall curve

    print("Precision-Recall: {:.2f}".format(metrics.auc(rec, prec)))

    class_names = ['0', '1']
    disp = plot_confusion_matrix(clf, clf_X_test, clf_y_test, display_labels=class_names, cmap=plt.cm.Blues)
    disp.ax_.set_title("Confusion Matrix")
    plt.show()

    # plot ROC Curve 
    plt.plot(fpr, tpr, color=colour, label=label) 
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    # plot Precision-Recall Curve
    plt.plot(prec, rec, color=colour, label=label) 
    plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    return fpr, tpr, prec, rec


    # train the model
logreg_clf = LogisticRegression()
logreg_clf.fit(clf_X_train, clf_y_train)

clf_y_pred = logreg_clf.predict(clf_X_test)

# model accuracy
print('--------------WITHOUT SMOTE--------------')
logreg_fpr, logreg_tpr, logreg_prec, logreg_rec = evaluate_model(logreg_clf, clf_X_train, clf_y_train, 'orange', 'LogReg')

pickle.dump(logreg_clf, open('logreg_clf.sav', 'wb'))

coefficients = logreg_clf.coef_
# create a scatter plot
coefficients
# plt.scatter(range(len(coefficients)), coefficients)
# plt.show()

# train the model
logreg_clf = LogisticRegression()
logreg_clf.fit(X_train_smote, y_train_smote)

clf_y_pred = logreg_clf.predict(clf_X_test)

# model accuracy
print('--------------WITH SMOTE--------------')
logreg_sm_fpr, logreg_sm_tpr, logreg_sm_prec, logreg_sm_rec = evaluate_model(logreg_clf, X_train_smote, y_train_smote, 'orange', 'LogReg')

pickle.dump(logreg_clf, open('logreg_clf_sm.sav', 'wb'))