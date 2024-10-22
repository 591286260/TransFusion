import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score, confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

# Load feature vectors from a file
X = np.loadtxt('SampleFeature.csv', delimiter=',')

# Construct labels
y = np.concatenate((np.ones(len(X) // 2), np.zeros(len(X) // 2)))

# Create LightGBM classifier
clf = lgb.LGBMClassifier(
    learning_rate=0.0033,
)

# Perform 5-fold cross-validation and calculate performance metrics
skf = KFold(n_splits=5, shuffle=True, random_state=123)
precision_list = []
recall_list = []
f1_score_list = []
acc_list = []
mcc_list = []

# Create empty lists to store AUC and AUPR values for each fold
fold_aucs = []
fold_auprs = []

for train_idx, test_idx in skf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train model
    clf.fit(X_train, y_train)

    # Predict labels
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred = np.where(y_pred_prob > threshold, 1, 0)

    # Output performance metrics
    print(classification_report(y_test, y_pred))

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fold_aucs.append(roc_auc)

    # Calculate AUPR
    aupr = average_precision_score(y_test, y_pred_prob)
    fold_auprs.append(aupr)

    # Save predictions and actual labels
    np.save(f"Y_pre{len(precision_list)}.npy", y_pred_prob)
    np.save(f"Y_test{len(precision_list)}.npy", y_test)

    # Calculate performance metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1_score)
    acc_list.append(accuracy)
    mcc_list.append(mcc)

    # Output performance metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"SPEC: {specificity:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print(f"MCC: {mcc:.4f}")

# Output average performance metrics from 5-fold cross-validation
print(f"Average accuracy: {np.mean(acc_list):.4f}")
print(f"Average precision: {np.mean(precision_list):.4f}")
print(f"Average recall: {np.mean(recall_list):.4f}")
print(f"Average spec: {np.mean(spec_list):.4f}")
print(f"Average f1-score: {np.mean(f1_score_list):.4f}")
print(f"Average MCC: {np.mean(mcc_list):.4f}")
