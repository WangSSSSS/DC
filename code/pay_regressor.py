import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.svm import SVR
from mlxtend.regressor import StackingRegressor


print('读取数据...')
pay_class = pd.read_csv('../Data/mid/pay_class_df.csv')
X2_price = pd.read_csv('../Data/mid/X2_price.csv')
Y2_price = pd.read_csv('../Data/mid/Y2_price.csv')
X_pred = pd.read_csv('../Data/mid/X_pred.csv')

user_id_pay0 = pay_class[pay_class['pay_class']==0].user_id.tolist()
user_id_pay = pay_class[pay_class['pay_class']==1].user_id.tolist()

X_pay_pred = X_pred[X_pred['user_id'].isin(user_id_pay)]


print('训练....')
X_train, X_test,y_train, y_test = train_test_split(X2_price, Y2_price, test_size=0.2, shuffle=True,random_state = 23)

# gbdt = GradientBoostingRegressor(loss='ls', alpha=0.9,n_estimators=500,learning_rate=0.05,max_depth=8,subsample=0.8,
#                                  min_samples_split=9,max_leaf_nodes=10)
# xgb = xgb.XGBRegressor(max_depth=5, n_estimators=500, learning_rate=0.05, eval_metric ='rmse',silent=False)
# lr = LinearRegression()
# rfg = RandomForestRegressor(bootstrap=False, max_features=0.05, min_samples_leaf=11, min_samples_split=8,n_estimators=100)
# svr_rbf = SVR(kernel='rbf')
#
# model = StackingRegressor(regressors=[gbdt, xgb, lr, rfg], meta_regressor=svr_rbf)
#
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print ("===> MSE:", metrics.mean_squared_error(y_test, y_pred))

other_params = {'num_boost_round':200,'learning_rate': 0.1, 'n_estimators': 30, 'max_depth': 5,
                'min_child_weight': 1, 'seed': 0,'subsample': 0.6, 'eval_metric': 'rmse','colsample_bytree': 0.8, 'gamma': 0.1,
                'reg_alpha': 2, 'reg_lambda': 1}
model = xgb.XGBRegressor(**other_params)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print ("===> MSE:",metrics.mean_squared_error(y_test, y_pred))


# model = RandomForestRegressor(bootstrap=False, max_features=0.05, min_samples_leaf=11, min_samples_split=8,n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print ("===> MSE:",metrics.mean_squared_error(y_test, y_pred))


y_pay_pred = model.predict(X_pay_pred)
sub_csv1 = pd.DataFrame({'user_id':user_id_pay,'prediction_pay_price':y_pay_pred})
sub_csv2 = pd.DataFrame({'user_id':user_id_pay0,'prediction_pay_price':[0]*len(user_id_pay0)})
sub_csv = pd.concat([sub_csv1 ,sub_csv2]).reset_index(drop=True)[['user_id','prediction_pay_price']]

sub_csv.to_csv('../Data/submit.csv',index=False)