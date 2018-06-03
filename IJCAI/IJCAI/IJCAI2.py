import pandas as pd
import time
# import lightgbm as lgb
from sklearn.metrics import log_loss
import numpy as np

X_train = np.load('./preprocess/X_train.npy')[()]
y_train = pd.read_csv('./preprocess/y_train.txt', header=None).values.ravel()
X_val = np.load('./preprocess/X_val.npy')[()]
y_val = pd.read_csv('./preprocess/y_val.txt', header=None).values.ravel()
X_test = np.load('./preprocess/X_test.npy')[()]
test_index = pd.read_csv('./preprocess/test_index.txt', header=None).values.ravel()



# gbm.fit(X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         eval_metric='binary_logloss',
#         early_stopping_rounds=150)

# gbm = lgb.LGBMRegressor(objective='binary',
#                         num_leaves=64,
#                         learning_rate=0.014,
#                         n_estimators=2000,
#                         colsample_bytree = 0.65,
#                         subsample = 0.75,
#                         max_depth=7,
#                         # reg_alpha = 0.4
#                         )

print (X_val.shape)
print (y_train.shape)
lgb = 1
exit()
gbm = lgb.LGBMRegressor(objective='binary',
                        num_leaves=63,
                        learning_rate=0.08,
                        n_estimators=1000,
                        colsample_bytree=0.65,
                        subsample=0.75,
                        max_depth=8,
                        # reg_alpha = 0.4
                        )

gbm.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        early_stopping_rounds=150)

print('Start predicting...')
# predict
y_pred_1 = gbm.predict(X_val, num_iteration=gbm.best_iteration_)
y_sub_1 = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

print("Logloss of lightgbm is ", log_loss(y_val, y_pred_1))


sub = pd.DataFrame()
sub['instance_id'] = list(test_index)

sub['instance_id'] = list(test_index)
sub['predicted_score'] = list(y_sub_1)
# sub.to_csv(str(learning_rate) + '_' + str(n_estimators) + '_round1_result_'+ time.strftime("%Y%m%d") + '.txt', sep=" ", index=False)
sub.to_csv('0.08_round1_result_'+ time.strftime("%Y%m%d") + '.txt', sep=" ", index=False)