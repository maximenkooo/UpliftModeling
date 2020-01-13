import pandas
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from lightgbm import LGBMClassifier
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

def uplift_fit_predict(model, X_train, treatment_train, target_train, X_test, return_model=True):
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
    if return_model:
        return predict_uplift, model_treatment, model_control
    else:
        return predict_uplift

def pickle_model(model_tr, model_c):
    name_file_model_tr = 'LGBM_model_tr_%s.pkl' %now
    name_file_model_tr = 'LGBM_model_c_%s.pkl' %now
    joblib.dump(model_tr, name_file_model_tr) 
    joblib.dump(model_c, name_file_model_c) 

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

def group_analisys(df_features, model,print_score=True,save_model=True,make_sibmition=True,
            filename='submission',return_score=False):
    """
    Вывод скора, создание сабмитов, сохранение модели
    """
    now = str(datetime.now()).replace(' ','-')[:13]
    indices_train = df_train.index
    indices_test = df_test.index
    indices_learn, indices_valid = train_test_split(df_train.index, test_size=0.3, random_state=123)

    # Оценка качества на валидации
    if print_score:
        valid_uplift = uplift_fit_predict(
            model=model,
            X_train=df_features.loc[indices_learn, :].fillna(0).values,
            treatment_train=df_train.loc[indices_learn, 'treatment_flg'].values,
            target_train=df_train.loc[indices_learn, 'target'].values,
            X_test=df_features.loc[indices_valid, :].fillna(0).values,
            return_model=False
            )
        valid_score = uplift_score(valid_uplift,
            treatment=df_train.loc[indices_valid, 'treatment_flg'].values,
            target=df_train.loc[indices_valid, 'target'].values,
            )
        print('Validation score:', valid_score)
        if return_score:
            return valid_score
    #
    if make_sibmition:
        test_uplift, model_treatment, model_control = uplift_fit_predict(
        model=model,
        X_train=df_features.loc[indices_train, :].fillna(0).values,
        treatment_train=df_train.loc[indices_train, 'treatment_flg'].values,
        target_train=df_train.loc[indices_train, 'target'].values,
        X_test=df_features.loc[indices_test, :].fillna(0).values,
        return_model=save_model
        )
        df_submission = pandas.DataFrame({'uplift': test_uplift}, index=df_test.index)
        name = filename + now + '.csv'
        df_submission.to_csv(name)
        print(f'File {name} is saved')
        if save_model:
            # Save the model as a pickle in a file 
            name_file_model_tr = 'LGBM_model_tr_%s.pkl' %now
            name_file_model_c = 'LGBM_model_c_%s.pkl' %now
            joblib.dump(model_treatment, name_file_model_tr) 
            joblib.dump(model_control, name_file_model_c)  
            print(f'Model {name_file_model_tr} is saved')
            print(f'Model {name_file_model_c} is saved')

# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.

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

def make_feats(group, index,logging=True):
    funs = ['mean','max','min','std']
    ser = pandas.DataFrame(index=index)
    for f in funs:
        if logging:
                ser = ser.join(np.log(group.agg(f)).replace([np.inf, -np.inf], 0).fillna(0).astype(int), rsuffix=f'_{f}')
        else:
            ser = ser.join(group.agg(f).fillna(0).astype(int), rsuffix=f'_{f}')
    return ser

def print_cv(df, pars, params_values_check={}):
    cv_res = {}
    for k, v in params_values_check.items():
        cv_res_small = {}
        for values_param in v:
            pars[k] = values_param
            model = LGBMClassifier(**pars)
            val_score = group_analisys(df,model=model,print_score=True,make_sibmition=False, return_score=True)
            cv_res_small[values_param] = val_score
        cv_res[k] = cv_res_small
    print(cv_res)

def add_fetures(df_clients, df_purchases):
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

    # кол-во покупок
    df_features = df_features.join(df_purchases.client_id.value_counts().rename('count_all_purschases'))
    print("Add feature count_all_purschases")
    #кол-во уникальных магазинов клиента
    df_features = df_features.join(df_purchases.groupby(['client_id'])['store_id'].nunique().rename('count_all_stores'))
    print("Add feature count_all_stores")

    # средняя, max, min u std цена покупок
    temp = df_purchases.groupby(['client_id','transaction_id']).purchase_sum.mean().rename('purch_sum').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature purch_sum X4")

    # среднее, max, min u std кол-во уникальных продуктов в разных магазинах
    temp = df_purchases.groupby(['client_id','store_id'])['product_id'].nunique().rename('product_uni').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature product_uni X4")

    #среднее, max, min u std кол-во товаров  за 1 покупку (транзакцию)
    temp = df_purchases.groupby(['client_id','transaction_id'])['product_quantity'].sum().rename('product_quant').groupby(['client_id'])
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature product_quant X4")

    # признаки трат и сбора экспрес и регулряный баллов по каждой транзакции
    # средняя, max, min u std цена покупок
    temp = df_purchases.groupby(['client_id','transaction_id']).regular_points_received.mean().rename('reg_rec').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index,logging=False))
    print("Add feature reg_rec X4")

    # средняя, max, min u std цена покупок
    temp = df_purchases.groupby(['client_id','transaction_id']).express_points_received.mean().rename('exp_rec').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index,logging=False))
    print("Add feature exp_rec X4")

    # средняя, max, min u std цена покупок
    temp = df_purchases.groupby(['client_id','transaction_id']).regular_points_spent.mean().rename('reg_spent').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index,logging=False))
    print("Add feature reg_spent X4")

    # средняя, max, min u std цена покупок
    temp = df_purchases.groupby(['client_id','transaction_id']).express_points_spent.mean().rename('exp_spent').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index,logging=False))
    print("Add feature exp_spent X4")

    # среднее, max, min u std кол-во уникальных продуктов за транзакцию
    temp = df_purchases.groupby(['client_id','transaction_id'])['product_id'].nunique().rename('uniq_prod_by_trunc').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature uniq_prod_by_trunc X4")

    temp = df_purchases.groupby(['client_id','transaction_id'])['trn_sum_from_iss'].mean().rename('trn_sum_from_iss').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature trn_sum_from_iss X4")

    temp = df_purchases.groupby(['client_id','transaction_id'])['trn_sum_from_red'].mean().rename('trn_sum_from_red').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature trn_sum_from_red X4")

    # для последнего месяца!!
    last_month = df_purchases[df_purchases['transaction_datetime'].astype(str) > '2019-02-18']

    # кол-во покупок
    df_features = df_features.join(last_month.client_id.value_counts().rename('last_count_all_purschases'))
    print("Add feature last_count_all_purschases")

    #кол-во уникальных магазинов клиента
    df_features = df_features.join(last_month.groupby(['client_id'])['store_id'].nunique().rename(
        'last_count_all_stores'))
    print("Add feature last_count_all_stores")

    # средняя, max, min u std цена покупок
    temp = last_month.groupby(['client_id','transaction_id']).purchase_sum.mean().rename(
        'last_purch_sum').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature last_purch_sum")

    # среднее, max, min u std кол-во уникальных продуктов в разных магазинах
    temp = last_month.groupby(['client_id','store_id'])['product_id'].nunique().rename(
        'last_product_uni').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature last_product_uni")

    #среднее, max, min u std кол-во товаров  за 1 покупку (транзакцию)
    temp = last_month.groupby(['client_id','transaction_id'])['product_quantity'].sum().rename(
        'last_product_quant').groupby(['client_id'])
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature last_product_quant  X4")

    # признаки трат и сбора экспрес и регулряный баллов по каждой транзакции
    # средняя, max, min u std цена покупок
    temp = last_month.groupby(['client_id','transaction_id']).regular_points_received.mean().rename(
        'last_reg_rec').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index,logging=False))
    print("Add feature last_reg_rec  X4")

    # средняя, max, min u std цена покупок
    temp = last_month.groupby(['client_id','transaction_id']).express_points_received.mean().rename(
        'last_exp_rec').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index,logging=False))
    print("Add feature last_exp_rec  X4")

    # средняя, max, min u std цена покупок
    temp = last_month.groupby(['client_id','transaction_id']).regular_points_spent.mean().rename(
        'last_reg_spent').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index,logging=False))
    print("Add feature last_reg_spent  X4")

    # средняя, max, min u std цена покупок
    temp = last_month.groupby(['client_id','transaction_id']).express_points_spent.mean().rename(
        'last_exp_spent').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index,logging=False))
    print("Add feature last_exp_spent  X4")

    # среднее, max, min u std кол-во уникальных продуктов за транзакцию
    temp = last_month.groupby(['client_id','transaction_id'])['product_id'].nunique().rename(
        'last_uniq_prod_by_trunc').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature last_uniq_prod_by_trunc  X4")

    temp = last_month.groupby(['client_id','transaction_id'])['trn_sum_from_iss'].mean().rename(
        'last_trn_sum_from_iss').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature last_trn_sum_from_iss  X4")

    temp = last_month.groupby(['client_id','transaction_id'])['trn_sum_from_red'].mean().rename(
        'last_trn_sum_from_red').groupby('client_id')
    df_features = df_features.join(make_feats(temp,df_features.index))
    print("Add feature last_trn_sum_from_red  X4")

    df_clients.first_issue_date = pandas.to_datetime(df_clients.first_issue_date)
    df_clients.first_redeem_date = pandas.to_datetime(df_clients.first_redeem_date)

    df_features['first_issue_day'] = df_clients.first_issue_date.apply(lambda x: x.day)
    df_features['first_issue_month'] = df_clients.first_issue_date.apply(lambda x: x.month)
    df_features['first_issue_year'] = df_clients.first_issue_date.apply(lambda x: x.year)
    print("Add feature first_issue_day,first_issue_month,first_issue_year")

    df_features['first_redeem_day'] = df_clients.first_redeem_date.apply(lambda x: x.day)
    df_features['first_redeem_month'] = df_clients.first_redeem_date.apply(lambda x: x.month)
    df_features['first_redeem_year'] = df_clients.first_redeem_date.apply(lambda x: x.year)
    print("Add feature first_redeem_day,first_redeem_month,first_redeem_year")

    lovely_product_df = (df_purchases.groupby(['client_id','product_id'])['product_id'].agg(['count']
    ).sort_values(by='count', ascending=False).reset_index().drop_duplicates('client_id', keep='first')
                    ).set_index('client_id')['count'].rename('lovely_prod_count')
    df_features = df_features.join(lovely_product_df)
    print("Add feature lovely_prod_count")

    lovely_product_df_last_month = (last_month.groupby(['client_id','product_id'])['product_id'].agg(['count']
    ).sort_values(by='count', ascending=False).reset_index().drop_duplicates('client_id', keep='first')
                    ).set_index('client_id')['count'].rename('last_lovely_prod_count')
    df_features = df_features.join(lovely_product_df_last_month)
    print("Add feature last_lovely_prod_count")

    imp_feats = ['first_issue_month', 'reg_spent_std', 'last_lovely_prod_count', 'product_quant_max', 'last_product_uni_max', 'trn_sum_from_iss_max', 'gender_M', 'first_redeem_day', 'gender_U', 'last_uniq_prod_by_trunc_max', 'count_all_purschases', 'last_purch_sum_max', 'last_count_all_stores', 'last_reg_rec_max', 'exp_rec_std', 'first_redeem_time', 'trn_sum_from_red_std', 'last_count_all_purschases', 'issue_redeem_delay', 'reg_rec_std', 'product_uni_max', 'uniq_prod_by_trunc_max', 'first_issue_day', 'uniq_prod_by_trunc_std', 'last_product_uni_std', 'age', 'purch_sum_max', 'last_trn_sum_from_iss_std', 'first_issue_year', 'first_redeem_month', 'count_all_stores', 'last_trn_sum_from_red_std', 'trn_sum_from_iss_std', 'last_trn_sum_from_red_max', 'product_uni_std', 'exp_rec_max', 'last_product_quant_std', 'lovely_prod_count', 'trn_sum_from_red_max', 'last_exp_spent_std', 'gender_F', 'last_uniq_prod_by_trunc_std', 'purch_sum_std', 'exp_spent_std', 'product_quant_std', 'last_product_quant_max', 'last_trn_sum_from_iss_max', 'first_redeem_year', 'last_reg_spent_max', 'reg_rec_max', 'last_reg_spent_std', 'last_purch_sum_std', 'first_issue_time']
    df_features = df_features[imp_feats]

    return df_features

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
    df_purchases = pandas.read_csv('data/purchases.csv')
    df_purchases = reduce_mem_usage(df_purchases)

    # Извлечение признаков

    df_features = add_fetures(df_clients, df_purchases)

    params = {'n_estimators':200,'learning_rate':0.03,'max_depth':4,'num_leaves':20,
             'min_data_in_leaf':3, 'application':'binary',
             'subsample':0.8, 'colsample_bytree': 0.8,
             'reg_alpha':0.01,'data_random_seed':42,'metric':'binary_logloss',
             'max_bin':416,'bagging_freq':3,'reg_lambda':0.01,'num_leaves':20             
    }
    model = LGBMClassifier(**params)
    group_analisys(df_features,model=model,print_score=True,make_sibmition=True, filename='submission')

    print('CV!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print_cv(df_features, params, params_values_check={'n_estimators':[i for i in range(100,1001,100)]})

