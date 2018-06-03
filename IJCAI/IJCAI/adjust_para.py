import lightgbm as lgb
import time
import pandas as pd
from hyperopt import hp,fmin,tpe,partial
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    return data

data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
data.drop_duplicates(inplace=True)
data = convert_data(data)
train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
            'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
            'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
            'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
            'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery',
            'shop_score_description',
            ]
target = ['is_trade']

def lgb_model(argsDict):
    num_leaves = argsDict["num_leaves"] + 56                   #56 - 70
    max_depth = argsDict["max_depth"] + 7                      #7 - 13
    n_estimators = argsDict['n_estimators'] * 5 + 40            #40 - 120
    learning_rate = argsDict["learning_rate"] * 0.01 + 0.05      #0.05 - 0.15

    print("max_depth:" + str(max_depth))
    print("n_estimator:" + str(n_estimators))
    print("learning_rate:" + str(learning_rate))

    gbm = lgb.LGBMClassifier(num_leaves=num_leaves,
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,)
    gbm.fit(train[features], train[target], feature_name=features,
            categorical_feature=['user_gender_id', ], eval_metric='binary_logloss')
    loss = log_loss(test[target], gbm.predict_proba(test[features], )[:, 1])
    print("loss:" + str(loss))
    return loss

def adjust_para():
    space = {"num_leaves": hp.randint("num_leaves", 14),
             "max_depth": hp.randint("max_depth", 6),
             "n_estimators": hp.randint("n_estimators", 16),
             "learning_rate": hp.randint("learning_rate", 10),
             }
    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(lgb_model, space, algo=algo, max_evals=100)
    min_loss = lgb_model(best)
    best_para = {"num_leaves": best["num_leaves"] + 56,
             "max_depth": best["max_depth"] + 7,
             "n_estimators": best['n_estimators'] * 5 + 40,
             "learning_rate": best["learning_rate"] * 0.01 + 0.05,
             }
    print("The best parameters are ", best_para)
    print("Minimum logloss is ", min_loss)

if __name__ == "__main__":
    adjust_para()
