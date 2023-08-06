# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn import metrics
# import lightgbm as lgbm
# from sklearn.preprocessing import OneHotEncoder, label_binarize
# from sklearn.metrics import accuracy_score, roc_auc_score
#
# # these were built datasets with reward and actions
# data_path = "C:/Users/logik/Desktop/rlsepsis234-master - full/data/"
# final_df_train = pd.read_csv(data_path + 'train_state_action_reward_df.csv')
#
# final_df_test = pd.read_csv(data_path+ 'test_state_action_reward_df.csv')
# final_df_train.drop('OutC002_90d mortality', axis=1, inplace=True)     # since we only predict terminal reward, it's same as predicting mortality
# final_df_test.drop('OutC002_90d mortality', axis=1, inplace=True)
# train_features = list(final_df_train.columns)[3:52]
# test_features = list(final_df_test.columns)[3:52]
# X_train = final_df_train[train_features]
# X_test = final_df_test[test_features]
# y_train = final_df_train['reward'].values.reshape(-1, 1)
# y_test = final_df_test['reward'].values.reshape(-1, 1)
# # print(y_test.value_counts().sort_values(ascending=False))
# from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from sklearn.metrics import make_scorer, roc_auc_score
# from sklearn.utils import class_weight
#
# # 计算类别权重
# sample_weights = class_weight.compute_sample_weight('balanced', y_train.ravel())
#
# # 参数网格
# param_grid = {
#     'learning_rate': [0.001],   # , 0.2, 0.1, 0.001
#     'num_leaves': [128],        # 30, 100
#     'n_estimators': [1024],    # 200, 2000
#     'max_depth': [16],
#     # 'min_child_samples': [50],
#     'min_data_in_leaf': [64],
#     'subsample': [1],        # , 0.6
#     'colsample_bytree': [1],     # , 0.6
#     'reg_alpha': [0.5],
#     'reg_lambda': [0.5]
#     }
# # param_grid = {
# #     'learning_rate': [0.04117742088984076],   # , 0.2, 0.1, 0.001
# #     'num_leaves': [34],        # 30, 100
# #     'n_estimators': [530],    # 200, 2000
# #     'max_depth': [8],
# #     # 'min_child_samples': [50],
# #     # 'min_data_in_leaf': [64],
# #     'subsample': [0.6475828586855414],        # , 0.6
# #     'colsample_bytree': [0.7296053366000121],     # , 0.6
# #     'reg_alpha': [0.28730715661804007],
# #     'reg_lambda': [0.42744674460519994]
# #     }
#
# # LGBM模型
# model = lgbm.LGBMClassifier(objective="multiclass", **param_grid)
#
# # 使用多分类的AUC作为评分标准
# multiclass_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovo")
#
# grid = GridSearchCV(model, param_grid, cv=5, scoring=multiclass_scorer)
# grid.fit(X_train, y_train.ravel(), sample_weight=sample_weights)
#
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
#
# # 使用最优模型进行预测
# best_model = grid.best_estimator_
# y_pred = best_model.predict(X_test)
# Y_pred_proba = best_model.predict_proba(X_test)
#
# # Convert y_test to one-hot encoded format for ROC AUC calculation
# y_test_one_hot = label_binarize(y_test, classes=[0, 1, 2])
#
# print('accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
# print('AUC-ovr: {0:0.4f}'.format(roc_auc_score(y_test_one_hot, Y_pred_proba, multi_class='ovr')))
# print('AUC-ovo: {0:0.4f}'.format(roc_auc_score(y_test_one_hot, Y_pred_proba, multi_class='ovo')))
#
# confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1, 2])
# cm_display.plot()
# plt.show()
#
# import joblib
#
# # Save the best model to a file
# model_save_path = "C:/Users/logik/Desktop/rlsepsis234-master - full/lightgbm_model.pkl"
# joblib.dump(best_model, model_save_path)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer
from sklearn.utils import class_weight

# Load data
data_path = "C:/Users/logik/Desktop/rlsepsis234-master - full/data/"
final_df_train = pd.read_csv(data_path + 'train_state_action_reward_df.csv')
final_df_test = pd.read_csv(data_path + 'test_state_action_reward_df.csv')
final_df_train.drop('OutC002_90d mortality', axis=1, inplace=True)
final_df_test.drop('OutC002_90d mortality', axis=1, inplace=True)
train_features = list(final_df_train.columns)[3:52]
test_features = list(final_df_test.columns)[3:52]
X_train = final_df_train[train_features]
X_test = final_df_test[test_features]
y_train = final_df_train['reward'].values.reshape(-1, 1)
y_test = final_df_test['reward'].values.reshape(-1, 1)

# Calculate class weights
class_weights = [np.sum(y_train == i) / len(y_train) for i in range(3)]
sample_weights_multiclass = np.array([class_weights[int(i)] for i in y_train.ravel()])
def fit(self, X, y=None, **fit_params):
    # Create a tqdm progress bar
    total_combinations = np.prod([len(v) for v in self.param_grid.values()])
    self._tqdm = tqdm(total=total_combinations, desc="Grid Search Progress")
    return super().fit(X, y, **fit_params)

# Parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01],
    'n_estimators': [500, 1000],
    'max_depth': [6, 10, 16],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'reg_alpha': [0.5, 1],
    'reg_lambda': [0.5, 1],
    'objective': ['multi:softmax'],
    'num_class': [3]
}

# XGBoost model
model = xgb.XGBClassifier(n_jobs=-1)

# Use multiclass AUC as the evaluation metric
multiclass_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovo")
from tqdm import tqdm

class GridSearchCVWithProgress(GridSearchCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y=None, **fit_params):
        # Create a tqdm progress bar
        total_combinations = np.prod([len(v) for v in self.param_grid.values()])
        self._tqdm = tqdm(total=total_combinations, desc="Grid Search Progress")
        return super().fit(X, y, **fit_params)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))

    def _evaluate_candidate(self, candidate_params):
        result = super()._evaluate_candidate(candidate_params)
        # Update the tqdm progress bar
        self._tqdm.update(1)
        return result

grid = GridSearchCVWithProgress(model, param_grid, cv=5, scoring=multiclass_scorer)
grid.fit(X_train, y_train.ravel(), sample_weight=sample_weights_multiclass)


print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Use the best model for predictions
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
Y_pred_proba = best_model.predict_proba(X_test)

# Convert y_test to one-hot encoded format for ROC AUC calculation
y_test_one_hot = label_binarize(y_test, classes=[0, 1, 2])

print('accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print('AUC-ovr: {0:0.4f}'.format(roc_auc_score(y_test_one_hot, Y_pred_proba, multi_class='ovr')))
print('AUC-ovo: {0:0.4f}'.format(roc_auc_score(y_test_one_hot, Y_pred_proba, multi_class='ovo')))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1, 2])
cm_display.plot()
plt.show()

import joblib

# Save the best model to a file
model_save_path = "C:/Users/logik/Desktop/rlsepsis234-master - full/xgboost_model.pkl"
joblib.dump(best_model, model_save_path)
