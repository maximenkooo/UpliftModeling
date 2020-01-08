import pandas
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score,KFold,cross_validate,train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier  
from datetime import datetime

def uplift_fit_predict(model, X_train, treatment_train, target_train, X_test):
    """
    Реализация простого способа построения uplift-модели.
    
    Обучаем два бинарных классификатора, которые оценивают вероятность target для клиента:
    1. с которым была произведена коммуникация (treatment=1)
    2. с которым не было коммуникации (treatment=0)
    
    В качестве оценки uplift для нового клиента берется разница оценок вероятностей:
    Predicted Uplift = P(target|treatment=1) - P(target|treatment=0)
    """
    X_treatment, y_treatment = X_train[treatment_train == 1, :], target_train[treatment_train == 1]
    X_control, y_control = X_train[treatment_train == 0, :], target_train[treatment_train == 0]
    model_treatment = clone(model).fit(X_treatment, y_treatment)
    model_control = clone(model).fit(X_control, y_control)
    predict_treatment = model_treatment.predict_proba(X_test)[:, 1]
    predict_control = model_control.predict_proba(X_test)[:, 1]
    predict_uplift = predict_treatment - predict_control
    return predict_uplift


def uplift_score(prediction, treatment, target, rate=0.3):
    """
    Подсчет Uplift Score
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()
    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score


def predict(df_features, model,print_score=True,make_sibmition=True, filename='submission'):
    
    indices_train = df_train.index
    indices_test = df_test.index
    indices_learn, indices_valid = train_test_split(df_train.index, test_size=0.3, random_state=123)

    valid_uplift = uplift_fit_predict(
        model=model,
        X_train=df_features.loc[indices_learn, :].fillna(0).values,
        treatment_train=df_train.loc[indices_learn, 'treatment_flg'].values,
        target_train=df_train.loc[indices_learn, 'target'].values,
        X_test=df_features.loc[indices_valid, :].fillna(0).values,
    )
    # Оценка качества на валидации

    if print_score:
        valid_score = uplift_score(valid_uplift,
        treatment=df_train.loc[indices_valid, 'treatment_flg'].values,
        target=df_train.loc[indices_valid, 'target'].values,
        )
        print('Validation score:', valid_score)
    if make_sibmition:
        test_uplift = uplift_fit_predict(
        model=model,
        X_train=df_features.loc[indices_train, :].fillna(0).values,
        treatment_train=df_train.loc[indices_train, 'treatment_flg'].values,
        target_train=df_train.loc[indices_train, 'target'].values,
        X_test=df_features.loc[indices_test, :].fillna(0).values,
        )

        df_submission = pandas.DataFrame({'uplift': test_uplift}, index=df_test.index)
        now = str(datetime.now()).replace(' ','-')[:13]
        df_submission.to_csv(filename + now + '.csv')

# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def make_feats(group, index):
    funs = ['mean','max','min','std']
    ser = pandas.DataFrame(index=index)
    for f in funs:
        ser = ser.join(group.agg(f).fillna(0).astype(int), lsuffix=f'_{f}',rsuffix=f'_{f}')
    return ser

def diff_betw_trans(df):
    dict_clietns_diffs_tranc = {}
    cli_uni = df.client_id.unique()
    for cli in cli_uni:
        client_df = df[df.client_id == cli].sort_values(by='transaction_datetime').drop_duplicates(
            subset='transaction_datetime')
        diff_day  = pandas.to_datetime(pandas.to_datetime(client_df.transaction_datetime).diff()).apply(
            lambda x : x.day)
        me = diff_day[1:].mean()
        dict_clietns_diffs_tranc[cli] = me
    return dict_clietns_diffs_tranc

if __name__ == "__main__":
    # Чтение данных
    df_clients = pandas.read_csv('data/clients.csv', index_col='client_id')
    # df_clients = reduce_mem_usage(df_clients)
    df_train = pandas.read_csv('data/uplift_train.csv', index_col='client_id')
    df_train = reduce_mem_usage(df_train)
    df_test = pandas.read_csv('data/uplift_test.csv', index_col='client_id')
    df_test = reduce_mem_usage(df_test)
    df_products = pandas.read_csv('data/products.csv', index_col='product_id')
    df_products = reduce_mem_usage(df_products)
    df_purchases = pandas.read_csv('data/purchases.csv', nrows=30000)
    df_purchases = reduce_mem_usage(df_purchases)

    # Извлечение признаков

    df_clients['first_issue_unixtime'] = pandas.to_datetime(df_clients['first_issue_date']).astype(int)/10**9
    df_clients['first_redeem_unixtime'] = pandas.to_datetime(df_clients['first_redeem_date']).astype(int)/10**9
    df_features = pandas.DataFrame({
        'gender_M': (df_clients['gender'] == 'M').astype(int),
        'gender_F': (df_clients['gender'] == 'F').astype(int),
        'gender_U': (df_clients['gender'] == 'U').astype(int),
        'age': df_clients['age'],
        'first_issue_time': df_clients['first_issue_unixtime'],
        'first_redeem_time': df_clients['first_redeem_unixtime'],
        'issue_redeem_delay': df_clients['first_redeem_unixtime'] - df_clients['first_issue_unixtime'],
    }).fillna(0)
    cols = df_features.columns

    # кол-во покупок
    df_features = df_features.join(df_purchases.client_id.value_counts().rename('count_all_purschases'))

    #кол-во уникальных магазинов клиента
    df_features = df_features.join(df_purchases.groupby(['client_id'])['store_id'].nunique().rename('count_all_stores'))

    # средняя, max, min u std цена покупок
    temp = df_purchases.groupby(['client_id','transaction_id']).purchase_sum.mean().rename('purch_sum_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # среднее, max, min u std кол-во уникальных продуктов в разных магазинах
    temp = df_purchases.groupby(['client_id','store_id'])['product_id'].nunique().rename('product_uni_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    #среднее, max, min u std кол-во товаров  за 1 покупку (транзакцию)
    temp = df_purchases.groupby(['client_id','transaction_id'])['product_quantity'].sum().rename('product_quant_').groupby(['client_id'])
    df_features = df_features.join(make_feats(temp,df_features.index))

    # признаки трат и сбора экспрес и регулряный баллов по каждой транзакции
    # средняя, max, min u std цена покупок
    temp = df_purchases.groupby(['client_id','transaction_id']).regular_points_received.mean().rename('reg_rec').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # средняя, max, min u std цена покупок
    temp = df_purchases.groupby(['client_id','transaction_id']).express_points_received.mean().rename('exp_rec_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # средняя, max, min u std цена покупок
    temp = df_purchases.groupby(['client_id','transaction_id']).regular_points_spent.mean().rename('reg_spent').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # средняя, max, min u std цена покупок
    temp = df_purchases.groupby(['client_id','transaction_id']).express_points_spent.mean().rename('exp_spent').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # среднее, max, min u std кол-во уникальных продуктов за транзакцию
    temp = df_purchases.groupby(['client_id','transaction_id'])['product_id'].nunique().rename('uniq_prod_by_trunc_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    temp = df_purchases.groupby(['client_id','transaction_id'])['trn_sum_from_iss'].mean().rename('trn_sum_from_iss_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    temp = df_purchases.groupby(['client_id','transaction_id'])['trn_sum_from_red'].mean().rename('trn_sum_from_red_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # для последнего месяца!!
    last_month = df_purchases[df_purchases['transaction_datetime'].astype(str) > '2019-02-18']

    # кол-во покупок
    df_features = df_features.join(last_month.client_id.value_counts().rename('last_count_all_purschases'))

    #кол-во уникальных магазинов клиента
    df_features = df_features.join(last_month.groupby(['client_id'])['store_id'].nunique().rename(
        'last_count_all_stores'))

    # средняя, max, min u std цена покупок
    temp = last_month.groupby(['client_id','transaction_id']).purchase_sum.mean().rename(
        'last_purch_sum_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # среднее, max, min u std кол-во уникальных продуктов в разных магазинах
    temp = last_month.groupby(['client_id','store_id'])['product_id'].nunique().rename(
        'last_product_uni_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    #среднее, max, min u std кол-во товаров  за 1 покупку (транзакцию)
    temp = last_month.groupby(['client_id','transaction_id'])['product_quantity'].sum().rename(
        'last_product_quant_').groupby(['client_id'])
    df_features = df_features.join(make_feats(temp,df_features.index))

    # признаки трат и сбора экспрес и регулряный баллов по каждой транзакции
    # средняя, max, min u std цена покупок
    temp = last_month.groupby(['client_id','transaction_id']).regular_points_received.mean().rename(
        'last_reg_rec').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # средняя, max, min u std цена покупок
    temp = last_month.groupby(['client_id','transaction_id']).express_points_received.mean().rename(
        'last_exp_rec_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # средняя, max, min u std цена покупок
    temp = last_month.groupby(['client_id','transaction_id']).regular_points_spent.mean().rename(
        'last_reg_spent').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # средняя, max, min u std цена покупок
    temp = last_month.groupby(['client_id','transaction_id']).express_points_spent.mean().rename(
        'last_exp_spent').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    # среднее, max, min u std кол-во уникальных продуктов за транзакцию
    temp = last_month.groupby(['client_id','transaction_id'])['product_id'].nunique().rename(
        'last_uniq_prod_by_trunc_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    temp = last_month.groupby(['client_id','transaction_id'])['trn_sum_from_iss'].mean().rename(
        'last_trn_sum_from_iss_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))

    temp = last_month.groupby(['client_id','transaction_id'])['trn_sum_from_red'].mean().rename(
        'last_trn_sum_from_red_').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))


    df_clients.first_issue_date = pandas.to_datetime(df_clients.first_issue_date)
    df_clients.first_redeem_date = pandas.to_datetime(df_clients.first_redeem_date)

    df_features['first_issue_day'] = df_clients.first_issue_date.apply(lambda x: x.day)
    df_features['first_issue_month'] = df_clients.first_issue_date.apply(lambda x: x.month)
    df_features['first_issue_year'] = df_clients.first_issue_date.apply(lambda x: x.year)

    df_features['first_redeem_day'] = df_clients.first_redeem_date.apply(lambda x: x.day)
    df_features['first_redeem_month'] = df_clients.first_redeem_date.apply(lambda x: x.month)
    df_features['first_redeem_year'] = df_clients.first_redeem_date.apply(lambda x: x.year)
        
    diffs = diff_betw_trans(df_purchases)
    df_features = df_features.join(pandas.Series(diffs,name='mean_diffs_tranc'))
    
    lovely_product_df = (df_purchases.groupby(['client_id','product_id'])['product_id'].agg(['count']
    ).sort_values(by='count', ascending=False).reset_index().drop_duplicates('client_id', keep='first')
                    ).set_index('client_id')['count'].rename('lovely_prod_count')
    df_features = df_features.join(lovely_product_df)

    lovely_product_df_last_month = (last_month.groupby(['client_id','product_id'])['product_id'].agg(['count']
    ).sort_values(by='count', ascending=False).reset_index().drop_duplicates('client_id', keep='first')
                    ).set_index('client_id')['count'].rename('last_lovely_prod_count')
    df_features = df_features.join(lovely_product_df_last_month)

#    df_features = reduce_mem_usage(df_features)

    params = {'n_estimators':200,'learning_rate':0.03,'max_depth':4,'num_leaves':20,
             'min_data_in_leaf':3, 'application':'binary',
             'subsample':0.8, 'colsample_bytree': 0.8,
             'reg_alpha':0.01,'data_random_seed':42,'metric':'binary_logloss',
             'max_bin':416,'bagging_freq':3,'reg_lambda':0.01,'num_leaves':20             
    }
    model = LGBMClassifier(**params)
    predict(df_features,model=model,print_score=True,make_sibmition=True, filename='submission')