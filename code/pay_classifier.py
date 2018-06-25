import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics

def read_data():
    print('读取数据')
    X1_class = pd.read_csv('../Data/mid/X1_class.csv')
    Y1_class = pd.read_csv('../Data/mid/Y1_class.csv')
    X_pred = pd.read_csv('../Data/mid/X_pred.csv')
    return X1_class, Y1_class, X_pred

def undersample():
    print('数据下采样')
    rus = RandomUnderSampler(random_state=6)
    x_resampled, y_resampled = rus.fit_sample(X1_class, Y1_class.values.ravel())
    return x_resampled, y_resampled

def train_data():
    X_train, X_test, y_train, y_test = train_test_split(
        x_resampled, y_resampled, test_size=0.3, random_state=0)
    print('开始训练....')
    model = XGBClassifier(
        learning_rate=0.1,
        n_estimators=140,
        max_depth=7,
        min_child_weight=3,
        gamma=0.3,
        subsample=0.9,
        colsample_bytree=0.6,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    model.fit(np.array(X_train), np.array(y_train))
    y_predict = model.predict(np.array(X_test))
    print("Accuracy : %.4f" % metrics.accuracy_score(y_test, y_predict))
    print('生成预测结果..........')
    y_predict = model.predict(np.array(X_pred))
    pay_class_df = pd.DataFrame({'user_id': X_pred['user_id'], 'pay_class': y_predict})[['user_id', 'pay_class']]
    pay_class_df.to_csv('../Data/mid/pay_class_df.csv', index=False)


if __name__ == '__main__':

    X1_class, Y1_class, X_pred = read_data()
    x_resampled, y_resampled = undersample()
    train_data()
    # X, Y1_class, Y2_price , X_pred = clean_data()
    print('完成!')

