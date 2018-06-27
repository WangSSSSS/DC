# Author: Ivan

import pandas as pd

def read_data():
#读取数据
    train_path ='../Data/tap_fun_train.csv'
    test_path = '../Data/tap_fun_test.csv'
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    all_df = [train_df,test_df]
#按照竞赛圈的帖子，尝试转了一下格式，不过貌似没有多大作用，可能因为这里大多都是数值型的吧。
    for df in all_df:
        df_int = df.select_dtypes(include=['int'])
        converted_int = df_int .apply(pd.to_numeric, downcast='unsigned')
        df_float = df.select_dtypes(include=['float'])
        converted_float = df_float.apply(pd.to_numeric, downcast='float')
        df[converted_int.columns] = converted_int
        df[converted_float.columns] = converted_float
    return train_df,test_df

def clean_data():
#数据处理和清洗
    concat_df = pd.concat([train_df,test_df]).reset_index(drop=True)
#生成一些时间特征，年月日周
    concat_df['register_time'] = pd.to_datetime(concat_df['register_time'])
    concat_df['year'] = concat_df['register_time'].dt.year
    concat_df['month']=concat_df['register_time'].dt.month
    concat_df['day']=concat_df['register_time'].dt.day
    concat_df['hour']=concat_df['register_time'].dt.hour
    concat_df['weekday']=concat_df['register_time'].dt.weekday
    del concat_df['register_time']
    
    
    names = ['wood','stone','ivory','meat','magic','infantry','cavalry','shaman','wound_infantry','wound_cavalry'
             ,'wound_shaman','general_acceleration','building_acceleration','reaserch_acceleration','training_acceleration'
             ]

          for c_name  in names:
            new  _c = c_name
            add_c = c_name +'_add_value'
            reduce_c = c_name +'_reduce_value'
            concat_df[new_c] = concat_df[add_c] - concat_df[reduce_c]
            
    concat_df['treatment_acceleration']=concat_df['treatment_acceleraion_add_value'] - \
                                        concat_df['treatment_acceleration_reduce_value']        
           
    reduce_columns = [x for x in concat_df.columns if x.endswith('reduce_value')]
    concat_df['reduce_column'] = 0
    for reduce_column in reduce_columns:
        concat_df['reduce_column']  = concat_df['reduce_column'] + concat_df[reduce_column]
    
    
        
    
# 支付过则 pay_class 为1  否则为0
    concat_df['pay_class'] = concat_df['prediction_pay_price'].apply(lambda x : 1 if x!=0 else 0)
# 数据清洗
    columns_list = concat_df.columns.tolist()
    columns_list.remove('pay_class')
    columns_list.remove('prediction_pay_price')
    n = 0
    for column in columns_list:
        if concat_df[concat_df[column] == 0].shape[0] / concat_df.shape[0] > 0.90:
            print('remove the column:{},missing_rate is :{:.2f}%'.format(
                column,concat_df[concat_df[column] == 0].shape[0]*100 /concat_df.shape[0]))
            columns_list.remove(column)
            n +=1
    print('total remove columns num:{}'.format(n))
    train_id = train_df.user_id.tolist()
    train_id_pay = train_df[train_df['prediction_pay_price']!=0].user_id.tolist()
    test_id = test_df.user_id.tolist()

    X1_class = concat_df[concat_df['user_id'].isin(train_id)][columns_list]

    Y1_class = concat_df[concat_df['user_id'].isin(train_id)][['pay_class']]

    X2_price = concat_df[concat_df['user_id'].isin(train_id_pay)][columns_list]

    Y2_price = concat_df[concat_df['user_id'].isin(train_id_pay)][['prediction_pay_price']]

    X_pred = concat_df[concat_df['user_id'].isin(test_id)][columns_list]
# 存储数据
    X1_class.to_csv('../Data/mid/X1_class.csv', index=False)
    X2_price.to_csv('../Data/mid/X2_price.csv', index=False)

    Y1_class.to_csv('../Data/mid/Y1_class.csv', index=False)
    Y2_price.to_csv('../Data/mid/Y2_price.csv', index=False)
    X_pred.to_csv('../Data/mid/X_pred.csv', index=False)
    # return X, Y1_class, Y2_price, X_pred

if __name__ == '__main__':
    print('读取数据...')
    train_df, test_df = read_data()
    print('处理与清洗数据')
    clean_data()
    # X, Y1_class, Y2_price , X_pred = clean_data()
    print('完成!')


