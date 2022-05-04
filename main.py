#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')  # 不显示警告

def data_pretreatment(data1):
    data = data1.copy()
    data['Distance'].fillna(-1, inplace = True)
    data['Discount_rate'].fillna(-1, inplace = True)
    data['Coupon_id'].fillna(-1, inplace = True)
    data['discount_rate'] = data['Discount_rate'].map(lambda x : (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / (float(str(x).split(':')[0])) if ':'  in str(x) else float(x))
    data['flag_of_manjian'] = data['Discount_rate'].map(lambda x : 1 if ':' in str(x) else 0)
    data['manjian_at_least_cost'] = data['Discount_rate'].map(lambda x : int(str(x).split(':')[0]) if ':'  in str(x) else -1)
    data['date_received'] = pd.to_datetime(data['Date_received'], format="%Y%m%d")
    columns = data.columns.tolist()
    if 'Date' in columns:
        data['date'] = pd.to_datetime(data['Date'], format = "%Y%m%d")
    data['weekday_receive'] = data['date_received'].map(lambda x: x.weekday())  # 星期几
    # data['is_weekend'] = data['weekday_receive'].map(lambda x: 1 if x == 6 or x == 7 else 0)  # 判断领券日是否为休息日
    data = pd.concat([data, pd.get_dummies(data['weekday_receive'], prefix='week')], axis=1)  # one-hot离散星期几
    # data['is_yuechu'] = data['date_received'].map(lambda x : -1 if pd.isnull(x) else 1 if x.day <= 10 and x.day > 0 else 0)  #判断月初
    # data['is_yuezhong'] = data['date_received'].map(lambda x : -1 if pd.isnull(x) else 1 if x.day <= 20 and x.day > 10 else 0)  #判断月中
    # data['is_yuemo'] = data['date_received'].map(lambda x : -1 if pd.isnull(x) else 1 if x.day <= 30 and x.day > 20 else 0)    #判断月末
    # data['is_Friday'] = data['weekday_receive'].map(lambda x : 1 if x == 5 else 0)
    return data

def online_data_pretreatment(data1):
    data = data1.copy()
    data['Discount_rate'].fillna(-1, inplace = True)
    data['Discount_rate'] = data['Discount_rate'].map(lambda x : 0 if str(x) == 'fixed' else x)
    data['discount_rate'] = data['Discount_rate'].map(lambda x : (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / (float(str(x).split(':')[0])) if ':'  in str(x) else float(x))
    data['flag_of_manjian'] = data['Discount_rate'].map(lambda x : 1 if ':' in str(x) else 0)
    data['is_fix'] = data['Coupon_id'].map(lambda x : 1 if str(x) == 'fixed' else 0)
    data['manjian_at_least_cost'] = data['Discount_rate'].map(lambda x : int(str(x).split(':')[0]) if ':'  in str(x) else -1)

    #将fixed转化成0号优惠卷
    data['Coupon_id'].fillna(-1, inplace = True)
    data['Coupon_id'] = data['Coupon_id'].map(lambda x : x if str(x) != 'fixed' else 0)

    #操作处理
    data['Action'].fillna(-1, inplace = True)
    #时间处理
    data['date_received'] = pd.to_datetime(data['Date_received'], format = '%Y%m%d')
    columns = data.columns.tolist()
    if 'Date' in columns:
        data['date'] = pd.to_datetime(data['Date'], format = '%Y%m%d')

    return data

def feat_prepare(history_field, label_field, keys, field):
    data = history_field.copy()
    data['Date'].fillna(-1, inplace=True)
    data['date'].fillna(-1, inplace=True)
    data['date_received'].fillna(-1, inplace=True)
    data['Date_received'].fillna(-1, inplace=True)
    data['cnt'] = 1
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    feat = label_field[keys].drop_duplicates(keep='first')

    return data, feat

def mer(data1, data2, key, fill):
    data1 = pd.merge(data1, data2, on = key, how = 'left')
    data1.fillna(fill, downcast='infer', inplace = True)
    return data1

def get_user_offline_featrue(history_field, label_field):       #1111111111111
    #主键和特征预处理
    keys = ['User_id']
    field = 'history_field'
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    data, u_feat = feat_prepare(history_field, label_field, keys, field)

    # 1.用户领卷数   1111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_cnt']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #2.用户领取优惠卷的最大距离   111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))].groupby(keys[0])['Distance'].max()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_max_distance']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #3.用户领取优惠卷的最小距离   111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))].groupby(keys[0])['Distance'].min()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_min_distance']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #4.用户领取优惠卷的平均距离  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))].groupby(keys[0])['Distance'].mean()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_mean_distance']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # 2.用户领卷并消费数      1111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_cnt']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # 3.用户领卷未消费数  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x == -1))].groupby(keys[0])['cnt'].count()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_not_consume_cnt']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # 4.用户领卷核销率   1111
    u_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_cnt'],u_feat[prefixs + 'receive_cnt']))

    # # 5.领取并消费优惠卷的平均折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1))].groupby(keys[0])['discount_rate'].mean()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_mean_discount_rate']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # # 6.领取并消费优惠卷平均距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1))].groupby(keys[0])['Distance'].mean()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_mean_Distance']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # # 7.在多少不同商家领取并消费优惠卷
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['Merchant_id'].apply(lambda x : len(set(x)))).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_differ_Merchant_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)

    # 8.在多少不同商家领取优惠卷    1111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))].groupby(keys[0])['Merchant_id'].apply(lambda x: len(set(x)))).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_differ_Merchant_cnt']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # # 9.在多少不同商家领取优惠卷核销率
    # u_feat[prefixs + 'receive_differ_Merchant_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_differ_Merchant_cnt'],u_feat[prefixs + 'receive_differ_Merchant_cnt']))
    #
    #
    # # 10.用户领取满减额度200以上优惠卷的数量
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['manjian_at_least_cost'].map(lambda x : x >= 200))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_200_manjian_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #11. 用户领取并核销满级额度200以上优惠卷的数量
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['manjian_at_least_cost'].map(lambda x : x >= 200)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_200_manjian_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #12. 用户满减额度200以上优惠卷核销率
    # u_feat[prefixs + 'receive_and_consume_200_manjian_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_200_manjian_cnt'], u_feat[prefixs + 'receive_200_manjian_cnt']))
    #
    # #13. 用户核销满减额度200以上优惠卷占所有核销优惠卷的比重
    # u_feat[prefixs + 'receive_and_consume_200_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_200_manjian_cnt'], u_feat[prefixs + 'receive_and_consume_cnt']))
    #
    #
    #
    # #14.用户核销优惠卷的最大折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))  & (data['discount_rate'].map(lambda x : x != -1))].groupby(keys[0])['discount_rate'].max()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_max_discount_rate']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # # 15.用户核销优惠卷的最小折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x != -1))].groupby(keys[0])['discount_rate'].min()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_min_discount_rate']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #16.用户核销过不同优惠卷的数量
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['Coupon_id'].apply(lambda x : len(set(x)))).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_differ_coupon_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    #17.用户领取不同优惠卷数量
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))].groupby(keys[0])['Coupon_id'].apply(lambda x: len(set(x)))).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_differ_coupon_cnt']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # #18.用户不同优惠卷核销率
    # u_feat[prefixs + 'receive_and_consume_differ_coupon_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_differ_coupon_cnt'], u_feat[prefixs + 'receive_differ_coupon_cnt']))
    #
    # #19.用户核销优惠卷中的最大商家距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Distance'].map(lambda x : x != -1))].groupby(keys[0])['Distance'].max()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_max_distance']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #20.用户核销优惠卷中最小商家距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Distance'].map(lambda x: x != -1))].groupby(keys[0])['Distance'].min()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_min_distance']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #21.用户领取且核销优惠卷距离小于5的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Distance'].map(lambda x : x <= 5)) & (data['Distance'].map(lambda x : x >= 0))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_distance_less_5_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #22.用户领取且核销优惠卷距离大于5的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Distance'].map(lambda x : x > 5)) & (data['Distance'].map(lambda x : x <= 10))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_distance_more_5_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #23.用户消费次数包括普通消费
    # tmp = pd.DataFrame(data[data['Date'].map(lambda x : x != -1)].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'common_consume_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #24.用户没有领取优惠卷但消费的次数
    # tmp = pd.DataFrame(data[((data['Coupon_id'].map(lambda x : x != -1)) | (data['Date_received'].map(lambda x : x == -1))) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'not_receive_but_consume_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #25.用户没有领取优惠卷但消费的次数占总消费次数的比重
    # u_feat[prefixs + 'not_receive_but_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'not_receive_but_consume_cnt'], u_feat[prefixs + 'common_consume_cnt']))
    #
    # #26.用户平均每个商家核销多少张优惠卷
    # u_feat[prefixs + 'receive_and_consume_mean_Merchant_cnt'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_cnt'], u_feat[prefixs + 'receive_differ_Merchant_cnt']))
    #
    # #27.用户领取优惠卷且核销的次数占全部消费的比重
    # u_feat[prefixs + 'receive_and_consume_in_common_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_cnt'], u_feat[prefixs + 'common_consume_cnt']))
    #
    # #28.用户领取优惠卷核销次数与用户领卷但不消费次数的比值
    # u_feat[prefixs + 'receive_and_consume_with_receive_not_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_cnt'], u_feat[prefixs + 'receive_not_consume_cnt']))
    #
    # #29.用户领取满减优惠卷的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['flag_of_manjian'].map(lambda x : x == 1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_manjian_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #30.用户领取满减卷的次数占总领卷次数的比重
    # u_feat[prefixs + 'receive_manjian_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_manjian_cnt'], u_feat[prefixs + 'receive_cnt']))
    #
    # #31.用户满减卷消费次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['flag_of_manjian'].map(lambda x : x == 1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_manjian_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    #
    # #32.用户领取满减卷且核销次数占全部满减卷的比重
    # u_feat[prefixs + 'receive_and_consume_manjian_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_manjian_cnt'], u_feat[prefixs + 'receive_manjian_cnt']))
    #
    # #33.用户核销满减卷占总核销次数的比重
    # u_feat[prefixs + 'receive_and_consume_manjian_in_all_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_manjian_cnt'], u_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #34.用户核销的优惠卷的中位数折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x != -1))].groupby(keys[0])['discount_rate'].apply(np.median)).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_median_discount_rate']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #35.用户核销优惠卷折扣率大于0.9的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x >= 0.9))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_more_0.9_discount_rate_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #36.用户领取优惠卷折扣率大于0.9的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))  & (data['discount_rate'].map(lambda x : x >= 0.9))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_more_0.9_discount_rate_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #37.用户折扣率大于0.9优惠卷的核销率
    # u_feat[prefixs + 'receive_and_consume_more_0.9_discount_rate_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_more_0.9_discount_rate_cnt'], u_feat[prefixs + 'receive_more_0.9_discount_rate_cnt']))
    #
    # #38.用户领取折扣率大于0.8且小于0.9的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x < 0.9)) & (data['discount_rate'].map(lambda x : x >= 0.8))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_less_0.9_more_0.8_discount_rate_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #39.用户核销折扣率大于0.8且小于0.9的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x < 0.9)) & (data['discount_rate'].map(lambda x : x >= 0.8))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_less_0.9_more_0.8_discount_rate_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #40.用户对折扣率大于0.8且小于0.9优惠卷核销率
    # u_feat[prefixs + 'receive_and_consume_less_0.9_more_0.8_discount_rate_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_less_0.9_more_0.8_discount_rate_cnt'], u_feat[prefixs + 'receive_less_0.9_more_0.8_discount_rate_cnt']))
    #
    # #41.用户领卷日期与消费日期之间的时间差
    data[prefixs + 'gap'] = list(map(lambda x, y : (x - y).days if x != -1 and y != -1 else 0, data['date'], data['date_received']))
    data[prefixs + 'is_consume_15day'] = list(map(lambda x : 1 if x >= 0 and x <= 15 else 0, data[prefixs + 'gap']))

    #42.用户领取优惠卷对特定商家15天内核销的次数    111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x : x == 1))].groupby(keys[0])['Merchant_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_cnt']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # #43.用户领取优惠卷后15天内核销的次数与核销优惠卷次数的比重
    # u_feat[prefixs + 'receive_and_consume_in15day_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_in15day_cnt'], u_feat[prefixs + 'receive_and_consume_cnt']))

    #44.用户领取优惠卷但15天内没有核销  1111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x : x == 0))].groupby(keys[0])['Merchant_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_not_consume_in15day_cnt']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # #45.用户15天内核销和15天内没核销的比值
    # u_feat[prefixs + 'receive_and_consume_in15day_with_not_consume_in15day_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_in15day_cnt'], u_feat[prefixs + 'receive_not_consume_in15day_cnt']))

    #46.用户领取优惠卷15天内的核销率 111
    u_feat[prefixs + 'receive_and_consume_in15day_with_all_consume_rate'] = list(map(lambda x, y : 0 if (x + y) == 0 else x / (x + y), u_feat[prefixs + 'receive_and_consume_in15day_cnt'], u_feat[prefixs + 'receive_not_consume_in15day_cnt']))
    #删掉第44个
    u_feat.drop(prefixs + 'receive_not_consume_in15day_cnt', axis = 1, inplace = True)

    #47.用户核销数与用户对领卷商家15天内的核销数的比重  1111
    u_feat[prefixs + 'receive_and_consume_with_Merhcant_consume_in15days_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_cnt'], u_feat[prefixs + 'receive_and_consume_in15day_cnt']))

    #48.用户15天内核销的不同优惠卷的数量     1111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x: x == 1)) & (data['discount_rate'].map(lambda x: x != -1))].groupby(keys[0])['Coupon_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_differ_Coupon_cnt']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #49.用户对不同卷在15天内的核销率    1111
    u_feat[prefixs + 'differ_Coupon_in15day_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'receive_and_consume_in15day_differ_Coupon_cnt'], u_feat[prefixs + 'receive_differ_coupon_cnt']))
    u_feat.drop(prefixs + 'receive_differ_coupon_cnt', axis = 1, inplace = True)
    #47.用户15天内核销的中位数折扣率  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x : x == 1)) & (data['discount_rate'].map(lambda x : x != -1))].groupby(keys[0])['discount_rate'].apply(np.median)).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_median_discount_rate']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #48.用户15天内核销的最大折扣率  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x : x == 1)) & (data['discount_rate'].map(lambda x : x != -1))].groupby(keys[0])['discount_rate'].max()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_max_discount_rate']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #49.用户15天内核销的最小折扣率  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x: x == 1)) & (data['discount_rate'].map(lambda x: x != -1))].groupby(keys[0])['discount_rate'].min()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_min_discount_rate']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # #50.用户15天内核销的平均距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x : x == 1)) & (data['Distance'].map(lambda x : x != -1))].groupby(keys[0])['Distance'].mean()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_mean_Distance']
    # u_feat = mer(u_feat, tmp, keys[0], 0)

    #51.用户15天内核销的最大距离    111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x : x == 1)) & (data['Distance'].map(lambda x : x != -1))].groupby(keys[0])['Distance'].max()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_max_Distance']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #52.用户15天内核销的平均距离   111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x: x == 1)) & (data['Distance'].map(lambda x: x != -1))].groupby(keys[0])['Distance'].mean()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_mean_Distance']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #53.用户15天内核销的最小距离 111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'is_consume_15day'].map(lambda x: x == 1)) & (data['Distance'].map(lambda x: x != -1))].groupby(keys[0])['Distance'].min()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_min_Distance']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #53.用户领卷日期和消费日期的平均间隙   111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data[prefixs + 'gap'].map(lambda x : x >= 0))].groupby(keys[0])[prefixs + 'gap'].mean()).reset_index()
    tmp.columns = [keys[0], prefixs + 'mean_gap']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    #54.用户在15天内核销的最小间隔    111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data[prefixs + 'gap'].map(lambda x: x >= 0))].groupby(keys[0])[prefixs + 'gap'].min()).reset_index()
    tmp.columns = [keys[0], prefixs + 'min_gap']
    u_feat = mer(u_feat, tmp, keys[0], 0)

    # #54.平均间隙标准化
    # u_feat[prefixs + 'mean_gap_01_standard'] = u_feat[prefixs + 'mean_gap'] / 30

    # #55.用户在周末领卷次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['is_weekend'].map(lambda x : x == 1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'weekend_receive_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #56.用户在周末核销次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['is_weekend'].map(lambda x : x == 1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'weekend_receive_and_consume_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #57.用户在周末核销率
    # u_feat[prefixs + 'weekend_receive_and_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'weekend_receive_and_consume_cnt'], u_feat[prefixs + 'weekend_receive_cnt']))
    #
    # #58.用户在周末核销次数占总核销次数的比重
    # u_feat[prefixs + 'weekend_receive_and_consume_with_all_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'weekend_receive_and_consume_cnt'], u_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #59.用户在月初领卷的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['is_yuechu'].map(lambda x : x == 1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'yuechu_receive_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #60.用户在月中领卷的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['is_yuezhong'].map(lambda x : x == 1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'yuezhong_receive_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #61.用户在月末领卷的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['is_yuemo'].map(lambda x : x == 1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'yuemo_receive_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #62.用户在月初核销的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['is_yuechu'].map(lambda x : x == 1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'yuechu_receive_and_consume_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #63.用户在月中核销的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['is_yuezhong'].map(lambda x: x == 1)) & (data['Date'].map(lambda x: x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'yuezhong_receive_and_consume_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #64.用户在月末核销的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['is_yuemo'].map(lambda x: x == 1)) & (data['Date'].map(lambda x: x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'yuemo_receive_and_consume_cnt']
    # u_feat = mer(u_feat, tmp, keys[0], 0)
    #
    # #65.用户在月初核销率
    # u_feat[prefixs + 'yuechu_receive_and_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, u_feat[prefixs + 'yuechu_receive_and_consume_cnt'], u_feat[prefixs + 'yuechu_receive_cnt']))
    #
    # #66.用户在月中核销率
    # u_feat[prefixs + 'yuezhong_receive_and_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, u_feat[prefixs + 'yuezhong_receive_and_consume_cnt'],u_feat[prefixs + 'yuezhong_receive_cnt']))
    #
    # #67.用户在月末核销率
    # u_feat[prefixs + 'yuemo_receive_and_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, u_feat[prefixs + 'yuemo_receive_and_consume_cnt'],u_feat[prefixs + 'yuemo_receive_cnt']))
    #
    # #68.用户在月初核销占总核销的比重
    # u_feat[prefixs + 'yuechu_receive_and_consume_with_all_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, u_feat[prefixs + 'yuechu_receive_and_consume_cnt'], u_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #69.用户在月中核销占总核销的比重
    # u_feat[prefixs + 'yuezhong_receive_and_consume_with_all_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, u_feat[prefixs + 'yuezhong_receive_and_consume_cnt'],u_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #70.用户在月末核销占总核销的比重
    # u_feat[prefixs + 'yuemo_receive_and_consume_with_all_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, u_feat[prefixs + 'yuemo_receive_and_consume_cnt'],u_feat[prefixs + 'receive_and_consume_cnt']))

    # #71.该用户在每天的领卷数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))].groupby(['User_id', 'date_received'])['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', 'date_received', prefixs + 'this_day_receive_cnt']
    # u_feat = mer(u_feat, tmp, ['User_id', 'date_received'], 0)
    #
    # #72.该用户在每天的核销数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'] != -1)].groupby(['User_id', 'date_received'])['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', 'date_received', prefixs + 'this_day_receive_and_consume_cnt']
    # u_feat = mer(u_feat, tmp, ['User_id', 'date_received'], 0)


    history_feat = label_field.copy()
    history_feat = pd.merge(history_feat, u_feat, on=keys, how='left')
    return history_feat

#商家特征群
def get_Merchant_featrue(history_field, label_field):       #11  11111111111
    # 主键和特征预处理
    keys = ['Merchant_id']
    field = 'history_field'
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    data, m_feat = feat_prepare(history_field, label_field, keys, field)

    #1.商家的优惠券被领取的次数   111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_cnt']
    m_feat = mer(m_feat, tmp, keys[0], 0)

    # #2.商家总共被消费次数
    # tmp = pd.DataFrame(data[data['Date'].map(lambda x : x != -1)].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'common_consume_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)


    #3.商家被不同客户领取的次数  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))].groupby(keys[0])['User_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_differ_User_cnt']
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #4.商家的券被核销的次数  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_cnt']
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #5.商家的券被核销率     111
    m_feat[prefixs + 'received_and_consumed_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y,m_feat[prefixs + 'receive_and_consume_cnt'],m_feat[prefixs + 'receive_cnt']))

    #6.商家的券没被核销的次数  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x == -1))].groupby(keys[0])['cnt'].count()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_not_consume_cnt']
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #7.商家提供的不同优惠券数  111
    tmp = pd.DataFrame(data[data['Coupon_id'].map(lambda x : x != -1)].groupby(keys[0])['Coupon_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = [keys[0], prefixs + 'differ_Coupon_cnt']
    m_feat = mer(m_feat, tmp, keys[0], 0)

    # #8.商店优惠卷核销平均折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x >= 0))].groupby(keys[0])['discount_rate'].mean()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_mean_discount_rate']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #9.商家优惠卷核销最大折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['discount_rate'].map(lambda x: x >= 0))].groupby(keys[0])['discount_rate'].max()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_max_discount_rate']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #10.商家优惠卷核销最小折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['discount_rate'].map(lambda x: x >= 0))].groupby(keys[0])['discount_rate'].min()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_min_discount_rate']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    #11.商家被不同客户核销的次数
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['User_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_differ_User_cnt']
    m_feat = mer(m_feat, tmp, keys[0], 0)

    # #12.不同用户核销商家优惠卷占领取的比重
    # m_feat[prefixs + 'receive_and_consume_differ_User_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, m_feat[prefixs + 'receive_and_consume_differ_User_cnt'], m_feat[prefixs + 'receive_differ_User_cnt']))

    #13.商家优惠卷平均每张被核销多少次  111
    m_feat[prefixs + 'Coupon_consume_mean_cnt'] = list(map(lambda x , y : 0 if y == 0 else x / y, m_feat[prefixs + 'receive_and_consume_cnt'], m_feat[prefixs + "differ_Coupon_cnt"]))

    #13.商家优惠卷平均每个用户核销多少张  111
    m_feat[prefixs + 'consume_mean_cnt'] = list(map(lambda x, y : 0 if y == 0 else x / y, m_feat[prefixs + 'receive_and_consume_cnt'], m_feat[prefixs + 'receive_and_consume_differ_User_cnt']))
    m_feat.drop(prefixs + 'receive_and_consume_differ_User_cnt', axis = 1, inplace = True)
    # #14.商家被核销过的不同优惠卷的数量
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['Coupon_id'].apply(lambda x: len(set(x)))).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_differ_coupon_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #15.商家领取过的所有不同优惠卷数量
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))].groupby(keys[0])['Coupon_id'].apply(lambda x: len(set(x)))).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_differ_coupon_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #16.商家被核销过不同优惠卷数量占所有领取过的不同优惠卷数量的比重
    # m_feat[prefixs + 'receive_and_consume_differ_coupon_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, m_feat[prefixs + 'receive_and_consume_differ_coupon_cnt'], m_feat[prefixs + 'receive_differ_coupon_cnt']))
    #
    # #17.商家平均每种优惠卷核销多少张
    # m_feat[prefixs + 'receive_and_consume_mean_coupon_cnt'] = list(map(lambda x, y : 0 if y == 0 else x / y, m_feat[prefixs + 'receive_and_consume_cnt'], m_feat[prefixs + 'receive_differ_coupon_cnt']))
    #
    #
    # #18.商家被核销优惠卷中平均距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Distance'].map(lambda x : x != -1))].groupby(keys[0])['Distance'].mean()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_mean_distance']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #19.商家被核销优惠卷最大距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Distance'].map(lambda x : x != -1))].groupby(keys[0])['Distance'].max()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_max_distance']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #20.商家被核销优惠卷最小距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Distance'].map(lambda x: x != -1))].groupby(keys[0])['Distance'].min()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_min_distance']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #21.商家被核销的优惠卷中距离小于5的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Distance'].map(lambda x : x >= 0)) & (data['Distance'].map(lambda x : x <= 5))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_distance_less_5_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    #
    # #22.商家总共被消费次数(包含普通消费)
    # tmp = pd.DataFrame(data[data['Date'].map(lambda x : x != -1)].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'common_consume_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #23.商家优惠卷核销折扣率在0.8-0.9的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x >= 0.8)) & (data['discount_rate'].map(lambda x : x <= 0.9))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_discount_rate_more_0.8_less_0.9_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #24.商家核销折扣率在0.8-0.9占总核销的比例
    # m_feat[prefixs + 'receive_and_consume_discount_rate_more_0.8_less_0.9_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, m_feat[prefixs + 'receive_and_consume_discount_rate_more_0.8_less_0.9_cnt'], m_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #25.商家优惠卷核销折扣率在0.9-1的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x >= 0.9)) & (data['discount_rate'].map(lambda x : x <= 1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_more_0.9_less_1_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #26.商家优惠卷核销的在0.9-1占总核销的比例
    # m_feat[prefixs + 'receive_and_consume_discount_rate_more_0.9_less_1_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, m_feat[prefixs + 'receive_and_consume_more_0.9_less_1_cnt'], m_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #27.商家优惠卷被领取后15天内被核销的次数
    data[prefixs + 'gap'] = list(map(lambda x, y: 0 if x == -1 or y == -1 else (x - y).days, data['date'], data['date_received']))
    data[prefixs + 'consume_in15day'] = list(map(lambda x: 1 if x >= 0 and x <= 15 else 0, data[prefixs + 'gap']))
    #
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data[prefixs + 'consume_in15day'].map(lambda x : x == 1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #28.商家优惠卷15天内被核销占总核销次数的比重
    # m_feat[prefixs + 'receive_and_consume_in15day_with_all_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, m_feat[prefixs + 'receive_and_consume_in15day_cnt'], m_feat[prefixs + 'receive_and_consume_cnt']))

    #29.商家优惠卷核销平均间隔天数 1111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data[prefixs + 'consume_in15day'].map(lambda x : x == 1))].groupby(keys[0])[prefixs + 'gap'].mean()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_mean_gap']
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #30.商家优惠卷核销最小间隔天数  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'consume_in15day'].map(lambda x: x == 1))].groupby(keys[0])[prefixs + 'gap'].min()).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_min_gap']
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #31.商家15天内核销的不同种类优惠卷数量
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'consume_in15day'].map(lambda x: x == 1))].groupby(keys[0])['Coupon_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_differ_Coupon_cnt']
    m_feat = mer(m_feat, tmp, keys[0], 0)

    #32.商家15天内核销不同优惠卷占所有优惠卷比重 1111
    m_feat[prefixs + 'consume_differ_coupon_in15day_with_all_coupon_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, m_feat[prefixs + 'receive_and_consume_in15day_differ_Coupon_cnt'], m_feat[prefixs + 'differ_Coupon_cnt']))

    # #30.商家优惠卷在周末领取且核销的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['is_weekend'].map(lambda x : x == 1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'weekend_receive_and_consume_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #31.商家周末核销优惠卷占总核销次数的比重
    # m_feat[prefixs + 'weekend_receive_and_consume_with_all_receive_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, m_feat[prefixs + 'weekend_receive_and_consume_cnt'], m_feat[prefixs + 'receive_and_consume_cnt']))
    #
    #
    # #32.商家在月初核销的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['is_yuechu'].map(lambda x : x == 1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'yuechu_receive_and_consume_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #33.商家在月中核销的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['is_yuezhong'].map(lambda x : x == 1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'yuezhong_receive_and_consume_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #34.商家在月末核销的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['is_yuemo'].map(lambda x: x == 1)) & (data['Date'].map(lambda x: x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'yuemo_receive_and_consume_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #35.商家在月初核销次数占总核销次数的比重
    # m_feat[prefixs + 'yuechu_receive_and_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, m_feat[prefixs + 'yuechu_receive_and_consume_cnt'], m_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #36.商家在月中核销次数占总核销次数的比重
    # m_feat[prefixs + 'yuezhong_receive_and_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, m_feat[prefixs + 'yuezhong_receive_and_consume_cnt'],m_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #37.商家在月末核销次数占总核销次数的比重
    # m_feat[prefixs + 'yuemo_receive_and_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, m_feat[prefixs + 'yuemo_receive_and_consume_cnt'],m_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #38.商家发布的满减卷的种数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['flag_of_manjian'].map(lambda x : x == 1))].groupby(keys[0])['Coupon_id'].apply(lambda x : len(set(x)))).reset_index()
    # tmp.columns = [keys[0], prefixs + 'differ_manjian_coupon_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #39.商家发布的折扣卷种类
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['flag_of_manjian'].map(lambda x : x != -1))].groupby(keys[0])['Coupon_id'].apply(lambda x: len(set(x)))).reset_index()
    # tmp.columns = [keys[0], prefixs + 'differ_zhekou_coupon_cnt']
    # m_feat = mer(m_feat, tmp, keys[0], 0)

    #40.商家15天内核销卷种类数量  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'consume_in15day'].map(lambda x: x == 1))].groupby(keys[0])['Coupon_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_differ_coupon_cnt']
    m_feat = mer(m_feat, tmp, keys[0], 0)

    # #41.商家15天内被核销优惠卷的最大距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'consume_in15day'].map(lambda x: x == 1) & (data['Distance'] != -1))].groupby(keys[0])['Distance'].max()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_max_distance']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #42.商家15天内核销优惠卷的平均距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'consume_in15day'].map(lambda x: x == 1) & (data['Distance'] != -1))].groupby(keys[0])['Distance'].mean()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_mean_distance']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #43.商家15天内核销优惠卷的最小距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data[prefixs + 'consume_in15day'].map(lambda x: x == 1) & (data['Distance'] != -1))].groupby(keys[0])['Distance'].min()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_in15day_min_distance']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #44.商家15天内核销优惠卷中位数折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['discount_rate'].map(lambda x: x >= 0)) & (data[prefixs + 'consume_in15day'].map(lambda x: x == 1))].groupby(keys[0])['discount_rate'].mean()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_mean_discount_rate']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #45.商家15天内核销优惠卷的最大折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['discount_rate'].map(lambda x: x >= 0)) & (data[prefixs + 'consume_in15day'].map(lambda x: x == 1))].groupby(keys[0])['discount_rate'].max()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_max_discount_rate']
    # m_feat = mer(m_feat, tmp, keys[0], 0)
    #
    # #46.商家15天内核销优惠卷的最小折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['discount_rate'].map(lambda x: x >= 0)) & (data[prefixs + 'consume_in15day'].map(lambda x: x == 1))].groupby(keys[0])['discount_rate'].min()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'receive_and_consume_min_discount_rate']
    # m_feat = mer(m_feat, tmp, keys[0], 0)

    m_feat.drop(prefixs + 'receive_and_consume_in15day_differ_Coupon_cnt', axis = 1, inplace = True)

    history_feat = label_field.copy()
    history_feat = pd.merge(history_feat, m_feat, on = keys, how = 'left')
    return history_feat

#优惠卷特征群
def get_Coupon_featrue(history_field, label_field):
    # 主键和特征预处理
    keys = ['Coupon_id']
    field = 'history_field'
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    data, c_feat = feat_prepare(history_field, label_field, keys, field)

    #1.优惠卷被领取的次数  111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))].groupby(keys[0])['cnt'].count()).reset_index()
    tmp.columns = [keys[0], prefixs + 'received_cnt']
    c_feat = mer(c_feat, tmp, keys[0], 0)

    #2.优惠卷15天内被核销次数  111
    tmp = pd.DataFrame(data[data['label'].map(lambda x : x == 1)].groupby(keys[0])['cnt'].count()).reset_index()
    tmp.columns = [keys[0], prefixs + 'received_and_consumed_cnt_15']
    c_feat = mer(c_feat, tmp, keys[0], 0)

    #3.优惠卷15天内未核销次数 111
    tmp = pd.DataFrame(data[data['label'].map(lambda x: x != 1)].groupby(keys[0])['cnt'].count()).reset_index()
    tmp.columns = [keys[0], prefixs + 'received_not_consumed_cnt_15']
    c_feat = mer(c_feat, tmp, keys[0], 0)

    #4.15天内核销/未核销  111
    c_feat[prefixs + 'consume_with_not_consume_rate_in15day'] = list(map(lambda x, y : 0 if y == 0 else x / y, c_feat[prefixs + 'received_and_consumed_cnt_15'], c_feat[prefixs + 'received_not_consumed_cnt_15']))

    #3.优惠卷15天内被核销率  111
    c_feat[prefixs + 'received_and_consumed_rate_15'] = list(map(lambda x, y : 0 if y == 0 else x / y, c_feat[prefixs + 'received_and_consumed_cnt_15'], c_feat[prefixs + 'received_cnt']))

    #5.优惠卷15天内未核销率  111
    c_feat[prefixs + 'received_not_consumed_rate_15'] = list(map(lambda x, y : 0 if y == 0 else x / y, c_feat[prefixs + 'received_not_consumed_cnt_15'], c_feat[prefixs + 'received_cnt']))

    #4.优惠卷15天内被核销的平均时间间隔  111
    data[prefixs + 'gap'] = list(map(lambda x, y: (x - y).days if x != -1 and y != -1 else 0, data['date'], data['date_received']))
    data[prefixs + 'is_consume_15day'] = list(map(lambda x: 1 if x >= 0 and x <= 15 else 0, data[prefixs + 'gap']))
    tmp1 = pd.DataFrame(data[(data[prefixs + 'is_consume_15day'] == 1)].groupby(keys[0])[prefixs + 'gap'].mean()).reset_index()
    tmp1.columns = [keys[0], prefixs + 'consumed_mean_time_gap_15']
    c_feat = mer(c_feat, tmp1, keys[0], 0)

    #5.满减优惠卷最低消费的中位数  111
    tmp = pd.DataFrame(data[(data['flag_of_manjian'].map(lambda x : x == 1)) & (data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))].groupby(keys[0])['manjian_at_least_cost'].apply(np.median)).reset_index()
    tmp.columns = [keys[0], prefixs + 'median_of_min_cost_of_manjian']
    c_feat = mer(c_feat, tmp, keys[0], 0)


    history_feat = label_field.copy()
    history_feat = pd.merge(history_feat, c_feat, on=keys, how='left')
    return history_feat

#用户商家交叉特征群
def get_user_Merchant_featrue(history_field, label_field):
    # 主键和特征预处理
    keys = ['User_id', 'Merchant_id']
    field = 'history_field'
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    data, uc_feat = feat_prepare(history_field, label_field, keys, field)

    #1.用户领取该商家优惠卷数目    111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1))].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'received_cnt']
    uc_feat = mer(uc_feat, tmp, keys, 0)

    #2.用户核销该商家的优惠卷数目     111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_cnt']
    uc_feat = mer(uc_feat, tmp, keys, 0)

    #3.用户领取该商家优惠卷后不核销次数    111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x == -1))].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_not_consume_cnt']
    uc_feat = mer(uc_feat, tmp, keys, 0)

    # #4.用户在该商家普通消费的次数
    # tmp = pd.DataFrame(data[((data['Coupon_id'].map(lambda x : x == -1)) | (data['Date_received'].map(lambda x : x == -1))) & (data['Date'].map(lambda x : x != -1))].groupby(keys)['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'common_consume_cnt']
    # uc_feat = mer(uc_feat, tmp, keys, 0)

    #5.用户领取商家核销率      111
    uc_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_consume_cnt'], uc_feat[prefixs + 'received_cnt']))

    #6.用户未核销率         1111
    uc_feat[prefixs + 'receive_not_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_not_consume_cnt'], uc_feat[prefixs + 'received_cnt']))


    # #6.用户对每个商家的不核销次数占用户总的不核销次数的比重
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x == -1))].groupby(keys)['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'not_consume_cnt']
    # uc_feat = mer(uc_feat, tmp, keys, 0)
    #
    # uc_feat[prefixs + 'receive_not_consume_every_merchant_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_not_consume_cnt'], uc_feat[prefixs + 'not_consume_cnt']))
    #
    # #7.用户对每个商家核销次数占用户总的核销次数的比重
    #
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1))].groupby('User_id')['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], prefixs + 'User_receive_and_consume_cnt']
    # uc_feat = mer(uc_feat, tmp, 'User_id', 0)
    #
    # uc_feat[prefixs + 'receive_and_consume_every_merchant_rate'] = list(map(lambda x , y : 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_consume_cnt'], uc_feat[prefixs + 'User_receive_and_consume_cnt']))
    #
    #
    # #8.用户对每个商家不核销次数占商家总的不核销次数率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x == -1))].groupby('Merchant_id')['cnt'].count()).reset_index()
    # tmp.columns = [keys[1], prefixs + 'Merchant_not_consume_cnt']
    # uc_feat = mer(uc_feat, tmp, 'Merchant_id', 0)
    #
    # uc_feat[prefixs + 'receive_not_consume_merchant_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_not_consume_cnt'], uc_feat[prefixs + 'Merchant_not_consume_cnt']))
    #
    # #9.用户对每个商家核销次数占商家总的核销次数的比重
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) &  (data['Date'].map(lambda x : x != -1))].groupby('Merchant_id')['cnt'].count()).reset_index()
    # tmp.columns = [keys[1], prefixs + 'Merchant_and_consume_cnt']
    # uc_feat = mer(uc_feat, tmp, 'Merchant_id', 0)
    #
    # uc_feat[prefixs + 'receive_and_consume_merchant_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_consume_cnt'],uc_feat[prefixs + 'Merchant_and_consume_cnt']))
    #
    # #10.用户核销商家平均距离
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1))].groupby(keys)['Distance'].mean()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_mean_distance']
    # uc_feat = mer(uc_feat, tmp, keys, 0)
    #
    # #11.用户核销商家距离小于5的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Distance'].map(lambda x : x <= 5)) & (data['Distance'].map(lambda x : x >= 0))].groupby(keys)['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_distance_less_5_cnt']
    # uc_feat = mer(uc_feat, tmp, keys, 0)
    #
    # #12.用户核销商家距离小于5次占该商家核销的比重
    # uc_feat[prefixs + 'consume_less_5_distance_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_consume_distance_less_5_cnt'], uc_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #13.用户核销商家的平均折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x != -1))].groupby(keys)['discount_rate'].mean()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_mean_discount_rate']
    # uc_feat = mer(uc_feat, tmp, keys, 0)
    #
    # #14.用户核销商家的最大折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x != -1))].groupby(keys)['discount_rate'].max()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_max_discount_rate']
    # uc_feat = mer(uc_feat, tmp, keys, 0)
    #
    # #15.用户核销商家的最小折扣率
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['discount_rate'].map(lambda x: x != -1))].groupby(keys)['discount_rate'].min()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_min_discount_rate']
    # uc_feat = mer(uc_feat, tmp, keys, 0)
    #
    # #16.用户核销商家的折扣率在0.9-1的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x >= 0.9)) & (data['discount_rate'].map(lambda x : x <= 1))].groupby(keys)['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_discount_rate_more_0.9_less_1_cnt']
    # uc_feat = mer(uc_feat, tmp, keys, 0)
    #
    # #17.用户核销折扣率在0.9-1的次数占总核销次数的比重
    # uc_feat[prefixs + 'receive_and_consume_discount_rate_more_0.9_less_1_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_consume_discount_rate_more_0.9_less_1_cnt'], uc_feat[prefixs + 'receive_and_consume_cnt']))
    #
    #
    # #18.用户核销商家折扣率在0.8到0.9的次数
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['discount_rate'].map(lambda x : x >= 0.8)) & (data['discount_rate'].map(lambda x : x <= 0.9))].groupby(keys)['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_discount_rate_more_0.8_less_0.9_cnt']
    # uc_feat = mer(uc_feat, tmp, keys, 0)
    #
    # #19.用户核销商家折扣率在0.8到0.9占比
    # uc_feat[prefixs + 'receive_and_consume_discount_rate_more_0.8_less_0.9_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y,uc_feat[prefixs + 'receive_and_consume_discount_rate_more_0.8_less_0.9_cnt'],uc_feat[prefixs + 'receive_and_consume_cnt']))
    #
    # #20.用户在该商家领到卷并15天内核销的次数
    # data[prefixs + 'gap'] = list(map(lambda x, y: 0 if x == -1 or y == -1 else (x - y).days, data['date'], data['date_received']))
    # data[prefixs + 'consume_in15day'] = list(map(lambda x: 1 if x >= 0 and x <= 15 else 0, data[prefixs + 'gap']))
    #
    # tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) & (data['Date'].map(lambda x : x != -1)) & (data[prefixs + 'consume_in15day'].map(lambda x : x == 1))].groupby(keys)['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_in15day_cnt']
    # uc_feat = mer(uc_feat, tmp, keys, 0)
    #
    # #21.用户领到卷并在15天内核销占该商家总核销比重
    # uc_feat[prefixs + 'receive_and_consume_in15day_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_consume_in15day_cnt'], uc_feat[prefixs + 'receive_and_consume_cnt']))


    history_feat = label_field.copy()
    history_feat = pd.merge(history_feat, uc_feat, on=keys, how='left')
    return history_feat

def get_user_coupon_featrue(history_field, label_field):
    # 主键和特征预处理
    keys = ['User_id', 'Coupon_id']
    field = 'history_field'
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    data, uc_feat = feat_prepare(history_field, label_field, keys, field)

    # 1.用户领取该商家优惠卷数目    111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'received_cnt']
    uc_feat = mer(uc_feat, tmp, keys, 0)

    # 2.用户核销该商家的优惠卷数目     111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x != -1))].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_cnt']
    uc_feat = mer(uc_feat, tmp, keys, 0)

    # 3.用户领取该商家优惠卷后不核销次数    111
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['Date'].map(lambda x: x == -1))].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_not_consume_cnt']
    uc_feat = mer(uc_feat, tmp, keys, 0)

    # #4.用户在该商家普通消费的次数
    # tmp = pd.DataFrame(data[((data['Coupon_id'].map(lambda x : x == -1)) | (data['Date_received'].map(lambda x : x == -1))) & (data['Date'].map(lambda x : x != -1))].groupby(keys)['cnt'].count()).reset_index()
    # tmp.columns = [keys[0], keys[1], prefixs + 'common_consume_cnt']
    # uc_feat = mer(uc_feat, tmp, keys, 0)

    # 5.用户领取商家核销率      111
    uc_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_consume_cnt'], uc_feat[prefixs + 'received_cnt']))

    # 6.用户未核销率         1111
    uc_feat[prefixs + 'receive_not_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, uc_feat[prefixs + 'receive_and_not_consume_cnt'], uc_feat[prefixs + 'received_cnt']))


    history_feat = label_field.copy()
    history_feat = pd.merge(history_feat, uc_feat, on=keys, how='left')
    return history_feat

def get_Merchant_Coupon_featrue(history_field, label_field):
    # 主键和特征预处理
    keys = ['Merchant_id', 'Coupon_id']
    field = 'history_field'
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    data, mc_feat = feat_prepare(history_field, label_field, keys, field)

    # 1.商家发放该优惠卷数量
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1))].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'received_cnt']
    mc_feat = mer(mc_feat, tmp, keys, 0)

    # 2.该商家该优惠卷15天内被核销数
    tmp = pd.DataFrame(data[data['label'] == 1].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_cnt_15']
    mc_feat = mer(mc_feat, tmp, keys, 0)

    #3.该商家该优惠卷核销率
    mc_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, mc_feat[prefixs + 'receive_and_consume_cnt_15'], mc_feat[prefixs + 'received_cnt']))

    history_feat = label_field.copy()
    history_feat = pd.merge(history_feat, mc_feat, on=keys, how='left')
    return history_feat

def get_User_discount_featrue(history_field, label_field):
    # 主键和特征预处理
    keys = ['User_id', 'discount_rate']
    field = 'history_field'
    prefixs = 'history_field_' + '_'.join(keys) + '_'
    data, ud_feat = feat_prepare(history_field, label_field, keys, field)

    # 1.用户领取该折扣优惠卷数量
    tmp = pd.DataFrame(data[(data['Coupon_id'].map(lambda x: x != -1)) & (data['Date_received'].map(lambda x: x != -1)) & (data['discount_rate'] != -1)].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'received_cnt']
    ud_feat = mer(ud_feat, tmp, keys, 0)

    # 2.用户该折扣率优惠卷15天内被核销数
    tmp = pd.DataFrame(data[(data['label'] == 1) & (data['discount_rate'] != -1)].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'receive_and_consume_cnt_15']
    ud_feat = mer(ud_feat, tmp, keys, 0)

    #3.用户对该折扣率优惠卷15天内未核销数
    tmp = pd.DataFrame(data[(data['label'] != 1) & (data['discount_rate'] != -1)].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = [keys[0], keys[1], prefixs + 'receive_not_consume_cnt_15']
    ud_feat = mer(ud_feat, tmp, keys, 0)

    #4.该商家该优惠卷核销率
    ud_feat[prefixs + 'receive_and_consume_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, ud_feat[prefixs + 'receive_and_consume_cnt_15'], ud_feat[prefixs + 'received_cnt']))

    history_feat = label_field.copy()
    history_feat = pd.merge(history_feat, ud_feat, on=keys, how='left')
    return history_feat


def get_leak_featrue(label_field):      #111111
    data = label_field.copy()
    t = label_field.columns.tolist()
    if 'Date' in t:
        data['Date'].fillna(-1, inplace=True)
        data['date'].fillna(-1, inplace=True)
    data['date_received'].fillna(-1, inplace=True)
    data['Date_received'].fillna(-1, inplace = True)
    data['Coupon_id'] = data['Coupon_id'].map(int)
    data['Date_received'] = data['Date_received'].map(int)
    data['cnt'] = 1

    feature = label_field.copy()
    data = data[data['Coupon_id'].map(lambda x : x != -1)]
    prefixs = 'leak_field_'


    #1.用户领取所有优惠卷的数目   111
    tmp = pd.DataFrame(data.groupby('User_id')['cnt'].count()).reset_index()
    tmp.columns = ['User_id', prefixs + 'User_receive_cnt']
    feature = mer(feature, tmp, 'User_id', 0)

    #2.用户领取满减卷数      111
    tmp = pd.DataFrame(data[data['flag_of_manjian'] == 1].groupby('User_id')['cnt'].count()).reset_index()
    tmp.columns = ['User_id', prefixs + 'User_receive_manjian_cnt']
    feature = mer(feature, tmp, 'User_id', 0)

    #3.用户领取非满减数      111
    tmp = pd.DataFrame(data[data['flag_of_manjian'] != 1].groupby('User_id')['cnt'].count()).reset_index()
    tmp.columns = ['User_id', prefixs + 'User_receive_not_manjian_cnt']
    feature = mer(feature, tmp, 'User_id', 0)

    #4.用户领取满减占比 111
    feature[prefixs + 'User_receive_manjian_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, feature[prefixs + 'User_receive_manjian_cnt'], feature[prefixs + 'User_receive_cnt']))

    #5.用户领取非满减优惠卷占比      111
    feature[prefixs + 'User_receive_notmanjian_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, feature[prefixs + 'User_receive_not_manjian_cnt'], feature[prefixs + 'User_receive_cnt']))

    #2.用户领取特定优惠卷的数目  111
    tmp = pd.DataFrame(data.groupby(['User_id', 'Coupon_id'])['cnt'].count()).reset_index()
    tmp.columns = ['User_id', 'Coupon_id', prefixs + 'User_Coupon_receive_cnt']
    feature = mer(feature, tmp, ['User_id', 'Coupon_id'], 0)

    # #3.用户领取特定优惠卷占总优惠卷的比重
    # feature[prefixs + 'User_Coupon_receive_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, feature[prefixs + 'User_Coupon_receive_cnt'], feature[prefixs + 'User_receive_cnt']))
    #
    # #4.用户领取优惠卷距离小于5的次数
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x : x >= 0)) & (data['Distance'].map(lambda x : x <= 5))].groupby('User_id')['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'User_receive_distance_less_5_cnt']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    # #5.用户领取优惠卷距离小于5占总领取次数的比重
    # feature[prefixs + 'receive_distance_less_5_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, feature[prefixs + 'User_receive_distance_less_5_cnt'], feature[prefixs + 'User_receive_cnt']))
    #
    # #6.用户领取优惠卷的平均距离
    # tmp = pd.DataFrame(data[data['Distance'].map(lambda x : x != -1)].groupby('User_id')['Distance'].mean()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'User_receive_mean_distance']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    # #7.用户领取优惠卷的最大距离
    # tmp = pd.DataFrame(data[data['Distance'].map(lambda x: x != -1)].groupby('User_id')['Distance'].max()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'User_receive_max_distance']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    # #8.用户领取优惠卷的最小距离
    # tmp = pd.DataFrame(data[data['Distance'].map(lambda x: x != -1)].groupby('User_id')['Distance'].min()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'User_receive_min_distance']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    # #9.用户领取优惠卷折扣率0.8-1的次数
    # tmp = pd.DataFrame(data[(data['discount_rate'].map(lambda x : x >= 0.8)) & (data['discount_rate'].map(lambda x : x <= 1))].groupby('User_id')['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'receive_discount_rate_more_0.8_less_1_cnt']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    # #10.用户领取优惠卷平均折扣率
    # tmp = pd.DataFrame(data.groupby('User_id')['discount_rate'].mean()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'User_receive_mean_discount_rate']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    # #11.用户领取优惠卷的最小折扣率
    # tmp = pd.DataFrame(data.groupby('User_id')['discount_rate'].min()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'User_receive_min_discount_rate']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    # #12.用户领取优惠卷的最大折扣率
    # tmp = pd.DataFrame(data.groupby('User_id')['discount_rate'].max()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'User_receive_max_discount_rate']
    # feature = mer(feature, tmp, 'User_id', 0)

    #13.用户领取特定商家的优惠卷数目  111
    tmp = pd.DataFrame(data.groupby(['User_id', 'Merchant_id'])['cnt'].count()).reset_index()
    tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_receive_merchant_cnt']
    feature = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)

    #14.用户领取特定商家满减卷数目 111
    tmp = pd.DataFrame(data[data['flag_of_manjian'] == 1].groupby(['User_id', 'Merchant_id'])['cnt'].count()).reset_index()
    tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_receive_merchant_manjian_cnt']
    feature = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)

    #15.用户领取特定商家非满减卷数目 111
    tmp = pd.DataFrame(data[data['flag_of_manjian'] != 1].groupby(['User_id', 'Merchant_id'])['cnt'].count()).reset_index()
    tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_receive_merchant_not_manjian_cnt']
    feature = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)

    #16.用户在该商家领满减占领卷数的比例 111
    feature[prefixs + 'User_receive_merchant_manjian_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, feature[prefixs + 'User_receive_merchant_manjian_cnt'], feature[prefixs + 'User_receive_merchant_cnt']))

    #17.用户在该商家领非满减占领卷数的比例  111
    feature[prefixs + 'User_receive_merchant_not_manjian_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, feature[prefixs + 'User_receive_merchant_not_manjian_cnt'], feature[prefixs + 'User_receive_merchant_cnt']))

    # #14.用户领取特定商家优惠卷的平均距离
    # tmp = pd.DataFrame(data[data['Distance'].map(lambda x : x != -1)].groupby(['User_id', 'Merchant_id'])['Distance'].mean()).reset_index()
    # tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_Merchant_receive_mean_distance']
    # feature = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)
    #
    # # 15.用户领取特定商家优惠卷的最大距离
    # tmp = pd.DataFrame(data[data['Distance'].map(lambda x: x != -1)].groupby(['User_id', 'Merchant_id'])['Distance'].max()).reset_index()
    # tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_Merchant_receive_max_distance']
    # feature = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)
    #
    # # 16.用户领取特定商家优惠卷的最小距离
    # tmp = pd.DataFrame(data[data['Distance'].map(lambda x: x != -1)].groupby(['User_id', 'Merchant_id'])['Distance'].min()).reset_index()
    # tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_Merchant_receive_min_distance']
    # feature = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)
    #
    # #17.用户领取特定商家优惠卷距离小于3的次数
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x : x >= 0)) & (data['Distance'].map(lambda x : x <= 3))].groupby(['User_id', 'Merchant_id'])['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_Merchant_receive_less_3_distance_cnt']
    # feature = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)
    #
    # # 18.用户领取特定商家优惠卷的平均折扣率
    # tmp = pd.DataFrame(data[data['discount_rate'].map(lambda x : x != -1)].groupby(['User_id', 'Merchant_id'])['discount_rate'].mean()).reset_index()
    # tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_Merchant_receive_mean_discount_rate']
    # feature = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)
    #
    # # 19.用户领取特定商家优惠卷的最大折扣率
    # tmp = pd.DataFrame(data[data['discount_rate'].map(lambda x: x != -1)].groupby(['User_id', 'Merchant_id'])['discount_rate'].max()).reset_index()
    # tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_Merchant_receive_max_discount_rate']
    # feature = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)
    #
    # # 20.用户领取特定商家优惠卷的最小折扣率
    # tmp = pd.DataFrame(data[data['discount_rate'].map(lambda x: x != -1)].groupby(['User_id', 'Merchant_id'])['discount_rate'].min()).reset_index()
    # tmp.columns = ['User_id', 'Merchant_id', prefixs + 'User_Merchant_receive_min_discount_rate']
    # feature = mer(feature, tmp, ['User_id', 'Merchant_id'], 0)
    #
    # #21.用户领取的不同商家数目
    # tmp = pd.DataFrame(data.groupby('User_id')['Merchant_id'].apply(lambda x : len(set(x)))).reset_index()
    # tmp.columns = ['User_id', prefixs + 'User_differ_Merchant_cnt']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    # #22.用户当天领取的优惠卷数目
    # tmp = pd.DataFrame(data.groupby(['User_id', 'Date_received'])['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', 'Date_received', prefixs + 'User_now_receive_cnt']
    # feature = mer(feature, tmp, ['User_id', 'Date_received'], 0)
    #
    # #23.用户当天领取特定优惠卷数目
    # tmp = pd.DataFrame(data.groupby(['User_id', 'Date_received', 'Coupon_id'])['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', 'Date_received', 'Coupon_id', prefixs + 'User_now_Coupon_receive_cnt']
    # feature = mer(feature, tmp, ['User_id', 'Date_received', 'Coupon_id'], 0)
    #
    # #24.用户领取的所有优惠卷种类的数目
    # tmp = pd.DataFrame(data.groupby('User_id')['Coupon_id'].count()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'User_Coupon_differ_cnt']
    # feature = mer(feature, tmp, 'User_id', 0)

    #25.商家被领取的优惠卷数目      1111
    tmp = pd.DataFrame(data.groupby('Merchant_id')['cnt'].count()).reset_index()
    tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_cnt']
    feature = mer(feature, tmp, 'Merchant_id', 0)

    #26.商家被领取满减卷数量      111
    tmp = pd.DataFrame(data[data['flag_of_manjian'] == 1].groupby('Merchant_id')['cnt'].count()).reset_index()
    tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_manjian_cnt']
    feature = mer(feature, tmp, 'Merchant_id', 0)

    #27.商家被领取的非满减卷的数量    1111
    tmp = pd.DataFrame(data[data['flag_of_manjian'] != 1].groupby('Merchant_id')['cnt'].count()).reset_index()
    tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_not_manjian_cnt']
    feature = mer(feature, tmp, 'Merchant_id', 0)

    #28.商家被领取满减卷的比重     111
    feature[prefixs + 'Merchant_receive_manjian_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, feature[prefixs + 'Merchant_receive_manjian_cnt'], feature[prefixs + 'Merchant_receive_cnt']))

    #29商家被领取非满减卷的比重    111
    feature[prefixs + 'Merchant_receive_not_manjian_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, feature[prefixs + 'Merchant_receive_not_manjian_cnt'], feature[prefixs + 'Merchant_receive_cnt']))


    # #26.商家被领取的特定优惠卷的数目
    # tmp = pd.DataFrame(data.groupby(['Merchant_id', 'Coupon_id'])['cnt'].count()).reset_index()
    # tmp.columns = ['Merchant_id', 'Coupon_id', prefixs + 'Merchant_Coupon_receive_cnt']
    # feature = mer(feature, tmp, ['Merchant_id', 'Coupon_id'], 0)
    #
    # #27.商家被多少不同用户领取的数目
    # tmp = pd.DataFrame(data.groupby('Merchant_id')['User_id'].apply(lambda x : len(set(x)))).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_Coupon_receive_cnt']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #28.商家发行的所有优惠卷种类数目
    # tmp = pd.DataFrame(data.groupby('Merchant_id')['Coupon_id'].apply(lambda x: len(set(x)))).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_Coupon_differ_receive_cnt']
    # feature = mer(feature, tmp, 'Merchant_id', 0)

    #29.用户是否第一次领取该商家的优惠卷 1111
    tmp = data.copy()
    tmp1 = tmp.sort_values(by = ['User_id', 'Merchant_id', 'Date_received']).reset_index().drop(columns = 'index', axis = 1)
    tmp1 = tmp1.drop_duplicates(subset = ['User_id', 'Merchant_id'], keep = 'first')
    tmp1 = tmp1[['User_id', 'Merchant_id', 'Date_received']]
    tmp1[prefixs + 'User_id_first_received_merchant_flag'] = 1
    feature = mer(feature, tmp1, ['User_id', 'Merchant_id', 'Date_received'], 0)

    #30.用户是否最后一次领取该商家优惠卷 1111
    tmp1 = tmp.sort_values(by=['User_id', 'Merchant_id', 'Date_received']).reset_index().drop(columns='index', axis=1)
    tmp1 = tmp1.drop_duplicates(subset=['User_id', 'Merchant_id'], keep='last')
    tmp1 = tmp1[['User_id', 'Merchant_id', 'Date_received']]
    tmp1[prefixs + 'User_id_last_received_merchant_flag'] = 1
    feature = mer(feature, tmp1, ['User_id', 'Merchant_id', 'Date_received'], 0)

    #30.用户是否第一次领取该优惠卷     111
    tmp1 = tmp.sort_values(by = ['User_id', 'Coupon_id', 'Date_received']).reset_index().drop(columns = 'index', axis = 1)
    tmp1 = tmp1.drop_duplicates(['User_id', 'Coupon_id'], keep = 'first')
    tmp1 = tmp1[['User_id', 'Coupon_id', 'Date_received']]
    tmp1[prefixs + 'User_id_first_receive_coupon_flag'] = 1
    feature = mer(feature, tmp1, ['User_id', 'Coupon_id', 'Date_received'], 0)

    #31.用户是否最后一次领取该优惠卷     1111
    tmp1 = tmp.sort_values(by=['User_id', 'Coupon_id', 'Date_received']).reset_index().drop(columns='index', axis=1)
    tmp1 = tmp1.drop_duplicates(['User_id', 'Coupon_id'], keep='last')
    tmp1 = tmp1[['User_id', 'Coupon_id', 'Date_received']]
    tmp1[prefixs + 'User_id_last_receive_coupon_flag'] = 1
    feature = mer(feature, tmp1, ['User_id', 'Coupon_id', 'Date_received'], 0)

    #31.用户是否第一次领取优惠卷     1111
    tmp1 = tmp.sort_values(by=['User_id', 'Date_received']).reset_index().drop(columns='index', axis=1)
    tmp1 = tmp1.drop_duplicates(['User_id'], keep='first')
    tmp1 = tmp1[['User_id', 'Date_received']]
    tmp1[prefixs + 'User_id_first_receive_flag'] = 1
    feature = mer(feature, tmp1, ['User_id', 'Date_received'], 0)

    #32.用户是否最后一次领取优惠卷    1111
    tmp1 = tmp.sort_values(by=['User_id', 'Date_received']).reset_index().drop(columns='index', axis=1)
    tmp1 = tmp1.drop_duplicates(['User_id'], keep='last')
    tmp1 = tmp1[['User_id', 'Date_received']]
    tmp1[prefixs + 'User_id_last_receive_flag'] = 1
    feature = mer(feature, tmp1, ['User_id', 'Date_received'], 0)

    # #31.用户在周末领优惠卷的数目
    # tmp = pd.DataFrame(data[(data['is_weekend'].map(lambda x : x == 1))].groupby('User_id')['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'weekend_User_receive_cnt']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    # #32.用户在周末领优惠卷占全部的比重
    # feature[prefixs + 'weekend_User_receive_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, feature[prefixs + 'weekend_User_receive_cnt'], feature[prefixs + 'User_receive_cnt']))
    #
    # #33.用户在月初领卷数目
    # tmp = pd.DataFrame(data[(data['is_yuechu'].map(lambda x : x == 1))].groupby('User_id')['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'yuechu_User_receive_cnt']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    # #34.用户在月中被领卷数目
    # tmp = pd.DataFrame(data[(data['is_yuezhong'].map(lambda x: x == 1))].groupby('User_id')['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'yuezhong_User_receive_cnt']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    #
    # #35.用户在月末领卷数目
    # tmp = pd.DataFrame(data[(data['is_yuemo'].map(lambda x: x == 1))].groupby('User_id')['cnt'].count()).reset_index()
    # tmp.columns = ['User_id', prefixs + 'yuemo_User_receive_cnt']
    # feature = mer(feature, tmp, 'User_id', 0)
    #
    #
    # #36.商家被领取的距离小于5的次数
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x : x >= 0)) & (data['Distance'].map(lambda x : x <= 5))].groupby('Merchant_id')['cnt'].count()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_distance_less_5_cnt']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #37.商家被领取的距离大于5的次数
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x : x >= 5)) & (data['Distance'].map(lambda x : x <= 10))].groupby('Merchant_id')['cnt'].count()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_distance_more_5_cnt']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #38.商家发行的优惠卷被领取的平均距离
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x : x != -1))].groupby('Merchant_id')['Distance'].mean()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_distance_mean']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #39.商家优惠卷中位数距离
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x: x != -1))].groupby('Merchant_id')['Distance'].apply(np.median)).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_distance_median']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #40.商家优惠卷最大距离
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x : x != -1))].groupby('Merchant_id')['Distance'].max()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_distance_max']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #41.商家优惠卷最小距离
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x: x != -1))].groupby('Merchant_id')['Distance'].min()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_distance_min']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #42.商家被领取优惠卷中位数折扣率
    # tmp = pd.DataFrame(data[(data['discount_rate'].map(lambda x : x != -1))].groupby('Merchant_id')['discount_rate'].apply(np.median)).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_discount_rate_median']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #43.商家被领取优惠卷平均折扣率
    # tmp = pd.DataFrame(data[(data['discount_rate'].map(lambda x: x != -1))].groupby('Merchant_id')['discount_rate'].mean()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_discount_rate_mean']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #44.商家被领取优惠卷最大折扣率
    # tmp = pd.DataFrame(data[(data['discount_rate'].map(lambda x: x != -1))].groupby('Merchant_id')['discount_rate'].max()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_discount_rate_max']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #45.商家被领取优惠卷最小折扣率
    # tmp = pd.DataFrame(data[(data['discount_rate'].map(lambda x: x != -1))].groupby('Merchant_id')['discount_rate'].min()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_discount_rate_min']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    #
    # #46.商家折扣率在0.8以上次数
    # tmp = pd.DataFrame(data[(data['discount_rate'].map(lambda x : x >= 0.8))].groupby('Merchant_id')['cnt'].count()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'Merchant_receive_discount_rate_more_0.8_cnt']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #47.商家优惠卷在周末被领取次数
    # tmp = pd.DataFrame(data[(data['is_weekend'].map(lambda x : x == 1))].groupby('Merchant_id')['cnt'].count()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'weekend_Merchant_id_receive_cnt']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #48.商家在月初领卷数目
    # tmp = pd.DataFrame(data[(data['is_yuechu'].map(lambda x : x == 1))].groupby('Merchant_id')['cnt'].count()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'yuechu_Merchant_receive_cnt']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #49.商家在月中被领卷数目
    # tmp = pd.DataFrame(data[(data['is_yuezhong'].map(lambda x: x == 1))].groupby('Merchant_id')['cnt'].count()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'yuezhong_Merchant_receive_cnt']
    # feature = mer(feature, tmp, 'Merchant_id', 0)
    #
    # #50.商家在月末领卷数目
    # tmp = pd.DataFrame(data[(data['is_yuemo'].map(lambda x: x == 1))].groupby('Merchant_id')['cnt'].count()).reset_index()
    # tmp.columns = ['Merchant_id', prefixs + 'yuemo_Merchant_receive_cnt']
    # feature = mer(feature, tmp, 'Merchant_id', 0)

    #51.优惠卷在该月被领取次数 1111
    tmp = pd.DataFrame(data.groupby('Coupon_id')['cnt'].count()).reset_index()
    tmp.columns = ['Coupon_id', prefixs + 'Coupon_receive_this_month_cnt']
    feature = mer(feature, tmp, 'Coupon_id', 0)

    #52.优惠卷被多少不同用户领取   111
    tmp = pd.DataFrame(data.groupby('Coupon_id')['User_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = ['Coupon_id', prefixs + 'Coupon_receive_this_month_differ_user_cnt']
    feature = mer(feature, tmp, 'Coupon_id', 0)

    #53.优惠卷被多少不同商家发放   111
    tmp = pd.DataFrame(data.groupby('Coupon_id')['Merchant_id'].apply(lambda x: len(set(x)))).reset_index()
    tmp.columns = ['Coupon_id', prefixs + 'Coupon_receive_this_month_differ_merchant_cnt']
    feature = mer(feature, tmp, 'Coupon_id', 0)

    # #53.优惠卷该月被领取的最大距离
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x : x != -1))].groupby('Coupon_id')['Distance'].max()).reset_index()
    # tmp.columns = ['Coupon_id', prefixs + 'Coupon_receive_this_month_max_distance']
    # feature = mer(feature, tmp, 'Coupon_id', 0)
    #
    # #54.优惠卷该月被领取的平均距离
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x: x != -1))].groupby('Coupon_id')['Distance'].mean()).reset_index()
    # tmp.columns = ['Coupon_id', prefixs + 'Coupon_receive_this_month_mean_distance']
    # feature = mer(feature, tmp, 'Coupon_id', 0)
    #
    # #55.优惠卷该月被领取的最小距离
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x: x != -1))].groupby('Coupon_id')['Distance'].min()).reset_index()
    # tmp.columns = ['Coupon_id', prefixs + 'Coupon_receive_this_month_min_distance']
    # feature = mer(feature, tmp, 'Coupon_id', 0)
    #
    # #56.优惠卷该月被领取的中位数距离
    # tmp = pd.DataFrame(data[(data['Distance'].map(lambda x: x != -1))].groupby('Coupon_id')['Distance'].apply(np.median)).reset_index()
    # tmp.columns = ['Coupon_id', prefixs + 'Coupon_receive_this_month_median_distance']
    # feature = mer(feature, tmp, 'Coupon_id', 0)

    #57.该折扣率优惠卷领取数     111
    tmp = pd.DataFrame(data[(data['discount_rate'] != -1)].groupby('discount_rate')['cnt'].count()).reset_index()
    tmp.columns = ['discount_rate', prefixs + 'discount_rate_receive_cnt']
    feature = mer(feature, tmp, 'discount_rate', 0)

    #58.该折扣率有多少种优惠卷     111
    tmp = pd.DataFrame(data[(data['discount_rate'] != -1)].groupby('discount_rate')['Coupon_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = ['discount_rate', prefixs + 'discount_rate_differ_coupon_cnt']
    feature = mer(feature, tmp, 'discount_rate', 0)

    #59.该折扣率优惠卷在多少不同商家被领取       111
    tmp = pd.DataFrame(data[(data['discount_rate'] != -1) & (data['date_received'] != -1)].groupby('discount_rate')['Merchant_id'].apply(lambda x: len(set(x)))).reset_index()
    tmp.columns = ['discount_rate', prefixs + 'discount_rate_differ_Merchant_cnt']
    feature = mer(feature, tmp, 'discount_rate', 0)

    #60.用户领取该优惠卷的数目     111
    tmp = pd.DataFrame(data[data['date_received'] != -1].groupby(['User_id', 'Coupon_id'])['cnt'].count()).reset_index()
    tmp.columns = ['User_id', 'Coupon_id', prefixs + 'User_receive_Coupon_cnt']
    feature = mer(feature, tmp, ['User_id', 'Coupon_id'], 0)

    #61.用户领取该折扣率优惠卷数量 111
    tmp = pd.DataFrame(data[data['date_received'] != -1].groupby(['User_id', 'discount_rate'])['cnt'].count()).reset_index()
    tmp.columns = ['User_id', 'discount_rate', prefixs + 'User_receive_discount_rate_cnt']
    feature = mer(feature, tmp, ['User_id', 'discount_rate'], 0)

    #62.用户领取该折扣不同优惠卷数    111
    tmp = pd.DataFrame(data[data['date_received'] != -1].groupby(['User_id', 'discount_rate'])['Coupon_id'].apply(lambda x : len(set(x)))).reset_index()
    tmp.columns = ['User_id', 'discount_rate', prefixs + 'User_receive_discount_rate_differ_coupon_cnt']
    feature = mer(feature, tmp, ['User_id', 'discount_rate'], 0)

    #63.用户在不同商家领取该折扣优惠卷数      111
    tmp = pd.DataFrame(data[data['date_received'] != -1].groupby(['User_id', 'discount_rate'])['Merchant_id'].apply(lambda x: len(set(x)))).reset_index()
    tmp.columns = ['User_id', 'discount_rate', prefixs + 'User_receive_discount_rate_differ_Merchant_cnt']
    feature = mer(feature, tmp, ['User_id', 'discount_rate'], 0)


    return feature

def get_online_featrue(online_field, label_field):
    # 主键和特征预处理
    keys = ['User_id']
    field = 'online_field'
    prefixs = 'online_field_' + '_'.join(keys) + '_'
    data, on_feat = feat_prepare(online_field, label_field, keys, field)

    #1.用户线上操作次数
    tmp = pd.DataFrame(data[data['Action'].map(lambda x : x != -1)].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = ['User_id', prefixs + 'User_action_cnt']
    on_feat = mer(on_feat, tmp, 'User_id', 0)


    #2.用户线上点击率
    tmp = pd.DataFrame(data[data['Action'].map(lambda x : x == 0)].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = ['User_id', prefixs + 'User_click_cnt']
    on_feat = mer(on_feat, tmp, 'User_id', 0)
    if prefixs + 'User_click_cnt' not in on_feat.columns.tolist():
        on_feat[prefixs + 'User_click_cnt'] = 0
    on_feat.fillna(0, downcast='infer', inplace = True)

    on_feat[prefixs + 'User_click_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, on_feat[prefixs + 'User_click_cnt'], on_feat[prefixs + 'User_action_cnt']))

    #3.用户线上购买率
    tmp = pd.DataFrame(data[data['Action'].map(lambda x: x == 1)].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = ['User_id', prefixs + 'User_buy_cnt']
    on_feat = mer(on_feat, tmp, 'User_id', 0)
    if prefixs + 'User_buy_cnt' not in on_feat.columns.tolist():
        on_feat[prefixs + 'User_buy_cnt'] = 0
    on_feat.fillna(0, downcast='infer', inplace=True)

    on_feat[prefixs + 'User_buy_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, on_feat[prefixs + 'User_buy_cnt'], on_feat[prefixs + 'User_action_cnt']))

    #4.用户线上领取率
    tmp = pd.DataFrame(data[data['Action'].map(lambda x: x == 2) | data['Date_received'].map(lambda x : x != -1)].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = ['User_id', prefixs + 'User_receive_cnt']
    on_feat = mer(on_feat, tmp, 'User_id', 0)
    if prefixs + 'User_receive_cnt' not in on_feat.columns.tolist():
        on_feat[prefixs + 'User_receive_cnt'] = 0
    on_feat.fillna(0, downcast='infer', inplace=True)

    on_feat[prefixs + 'User_receive_rate'] = list(map(lambda x, y: 0 if y == 0 else x / y, on_feat[prefixs + 'User_receive_cnt'], on_feat[prefixs + 'User_action_cnt']))

    #5.用户线上不消费次数
    tmp = pd.DataFrame(data[data['Date'].map(lambda x : x == -1)].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = ['User_id', prefixs + 'User_not_consume_cnt']
    on_feat = mer(on_feat, tmp, 'User_id', 0)

    #6.用户线上优惠卷核销次数
    tmp = pd.DataFrame(data[(data['Date'].map(lambda x : x != -1)) & (data['Date_received'].map(lambda x : x != -1)) | data['Action'].map(lambda x : x == 2)].groupby(keys)['cnt'].count()).reset_index()
    tmp.columns = ['User_id', prefixs + 'User_consume_cnt']
    on_feat = mer(on_feat, tmp, 'User_id', 0)

    #7.用户线上优惠卷核销率
    on_feat[prefixs + 'User_consume_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, on_feat[prefixs + 'User_consume_cnt'], on_feat[prefixs + 'User_receive_cnt']))


    history_feat = label_field.copy()
    history_feat = pd.merge(history_feat, on_feat, on=keys, how='left')

    return history_feat

def get_dataset(history_field, online_field, label_field):
    history_feat = get_user_offline_featrue(history_field, label_field)
    merchant_feat = get_Merchant_featrue(history_field, label_field)
    coupon_feat = get_Coupon_featrue(history_field, label_field)
    um_feat = get_user_Merchant_featrue(history_field, label_field)
    uc_feat = get_user_coupon_featrue(history_field, label_field)
    leak_feat = get_leak_featrue(label_field)
    online_feat = get_online_featrue(online_field, label_field)
    ud_feat = get_User_discount_featrue(history_field, label_field)
    mc_feat = get_Merchant_Coupon_featrue(history_field, label_field)

    share_characters = list(set(label_field.columns.tolist()) &
                            set(history_feat.columns.tolist()) &
                            set(merchant_feat.columns.tolist()) &
                            set(leak_feat.columns.tolist()) &
                            set(coupon_feat.columns.tolist()) &
                            set(um_feat.columns.tolist())   &
                            set(online_feat.columns.tolist()) &
                            set(uc_feat.columns.tolist())  &
                            set(ud_feat.columns.tolist())  &
                            set(mc_feat.columns.tolist())
                            )
    label_field.index = range(len(label_field))
    dataset = pd.concat([label_field, history_feat.drop(share_characters, axis=1)], axis=1)
    dataset = pd.concat([dataset, leak_feat.drop(share_characters, axis = 1)], axis = 1)
    dataset = pd.concat([dataset, merchant_feat.drop(share_characters, axis = 1)], axis = 1)
    dataset = pd.concat([dataset, coupon_feat.drop(share_characters, axis = 1)], axis = 1)
    dataset = pd.concat([dataset, um_feat.drop(share_characters, axis = 1)], axis = 1)
    dataset = pd.concat([dataset, online_feat.drop(share_characters, axis = 1)], axis = 1)
    dataset = pd.concat([dataset, uc_feat.drop(share_characters, axis=1)], axis=1)
    dataset = pd.concat([dataset, ud_feat.drop(share_characters, axis = 1)], axis = 1)
    dataset = pd.concat([dataset, mc_feat.drop(share_characters, axis = 1)], axis = 1)
    #1.dislike_rate
    prefixs1 = 'history_field_' + '_'.join(['User_id', 'Merchant_id']) + '_'
    prefixs2 = 'history_field_' + '_'.join(['User_id']) + '_'
    prefixs3 = 'history_field_' + '_'.join(['Merchant_id']) + '_'
    dataset['User_dislike_Merhcant_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, dataset[prefixs1 + 'receive_and_not_consume_cnt'], dataset[prefixs2 + 'receive_not_consume_cnt']))
    #2.like_rate
    dataset['User_like_Merchant_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, dataset[prefixs1 + 'receive_and_consume_cnt'], dataset[prefixs2 + 'receive_and_consume_cnt']))
    #3.Merchant_like_rate
    dataset['Merchant_User_like_rate'] = list(map(lambda x, y : 0 if y == 0 else x / y, dataset[prefixs1 + 'receive_and_consume_cnt'], dataset[prefixs3 + 'receive_and_consume_cnt']))

    if 'Date' in dataset.columns.tolist():
        dataset.drop(['Merchant_id', 'Discount_rate', 'Date','date_received', 'date'], axis=1, inplace=True)
        label = dataset['label'].tolist()
        dataset.drop(['label'], axis=1, inplace=True)
        dataset['label'] = label
    else:
        dataset.drop(['Merchant_id', 'Discount_rate','date_received'], axis=1, inplace=True)

    dataset['User_id'] = dataset['User_id'].map(int)
    dataset['Coupon_id'] = dataset['Coupon_id'].map(int)
    dataset['Date_received'] = dataset['Date_received'].map(int)
    dataset['Distance'] = dataset['Distance'].map(int)
    if 'label' in dataset.columns.tolist():
        dataset['label'] = dataset['label'].map(int)

    dataset.drop_duplicates(keep='first', inplace=True)
    dataset.index = range(len(dataset))
    return dataset

def get_label(data1):
    data = data1.copy()
    data['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0, data['date'], data['date_received']))
    return data

def model_xgb(train, test):
    params = {'booster': 'gbtree','objective': 'binary:logistic','eval_metric': 'auc','silent': 1,'eta': 0.01,'max_depth': 5,'min_child_weight': 1,
              'gamma': 0,'lambda': 1,'colsample_bylevel': 0.7,'colsample_bytree': 0.7,'subsample': 0.9,'scale_pos_weight': 1, 'tree_method' : 'gpu_hist', 'predictor' : 'gpu_predictor'}
    # params = {'booster' : 'gbtree',
    #           'objective' : 'binary:logistic',
    #           'eval_metric' : 'auc',
    #           'tree_method' : 'gpu_hist',
    #           'max_depth' : 2,
    #           'min_child_weight' : 1.04,
    #           'lambda' : 11,
    #           'gamma' : 0.22,
    #           'subsample' : 0.7,
    #           'colsample_bytree' : 0.763,
    #           'colsample_bylevel' : 0.71,
    #           'eta' : 0.1,
    #           'nthread' : 5,
    #           'predictor' : 'gpu_predictor',
    #           'verbosity' : 1
    #           }
    tr = train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis = 1)
    te = test.drop(['User_id', 'Coupon_id', 'Date_received'], axis = 1)
    train_data = xgb.DMatrix(tr, label=train['label'])
    test_data = xgb.DMatrix(te)
    w = [(train_data, 'train')]

    model = xgb.train(params, train_data, num_boost_round=5000, evals=w, early_stopping_rounds=50)
    predict = pd.DataFrame(model.predict(test_data, validate_features=False), columns = ['prob'])

    return predict

if __name__ == '__main__':
    train_data = pd.read_csv('ccf_offline_stage1_train.csv')
    test_data = pd.read_csv('ccf_offline_stage1_test_revised.csv')
    online_data = pd.read_csv('ccf_online_stage1_train.csv')
    print('数据预处理中')
    train_data = data_pretreatment(train_data)
    test_data = data_pretreatment(test_data)
    online_data = online_data_pretreatment(online_data)
    print('训练数据打标中')
    train_data = get_label(train_data)
    online_data = get_label(online_data)


    # # 划分区间
    # print('划分数据集1中')
    # train_history_field1 = train_data[train_data['date'].isin(pd.date_range('2016/1/1', periods=104)) |(train_data['date_received'].isin(pd.date_range('2016/1/1', periods=104)) & train_data['date'].map(lambda x : pd.isnull(x)))]                         # [20160101, 20160413]
    # train_label_field1 = train_data[train_data['date_received'].isin(pd.date_range('2016/4/14', periods=30))]                                                                                                                                                # [20160414, 20160514]
    #
    # print('划分数据集2中')
    # train_history_field2 = train_data[train_data['date'].isin(pd.date_range('2016/2/1', periods=104)) |(train_data['date_received'].isin(pd.date_range('2016/2/1', periods=104)) & train_data['date'].map(lambda x : pd.isnull(x)))]                         # [20160201, 20160514]
    # train_label_field2 = train_data[train_data['date_received'].isin(pd.date_range('2016/5/15', periods=30))]                                                                                                                                                # [20160515, 20160615]
    #
    # print('划分测试集中')
    # test_history_field = train_data[train_data['date'].isin(pd.date_range('2016/3/15', periods=108)) |(train_data['date_received'].isin(pd.date_range('2016/3/15', periods=108)) & train_data['date'].map(lambda x : pd.isnull(x)))]                         # [20160101, 20160413]
    # test_label_field = test_data.copy()                                                                                                                                                                                                                      # [20160701, 20160801]

    #划分区间
    print('划分数据集中')
    train_history_data = train_data[
        train_data['date_received'].isin(pd.date_range('2016/3/2', periods=60))]  # [20160302,20160501)
    train_middle_data = train_data[train_data['date'].isin(pd.date_range('2016/5/1', periods=15))]  # [20160501,20160516)
    train_label_data = train_data[
        train_data['date_received'].isin(pd.date_range('2016/5/16', periods=31))]  # [20160516,20160616)
    # 验证集历史区间、中间区间、标签区间
    validate_history_data = train_data[
        train_data['date_received'].isin(pd.date_range('2016/1/16', periods=60))]  # [20160116,20160316)
    validate_middle_data = train_data[
        train_data['date'].isin(pd.date_range('2016/3/16', periods=15))]  # [20160316,20160331)
    validate_label_data = train_data[
        train_data['date_received'].isin(pd.date_range('2016/3/31', periods=31))]  # [20160331,20160501)
    # 测试集历史区间、中间区间、标签区间
    test_history_data = train_data[
        train_data['date_received'].isin(pd.date_range('2016/4/17', periods=60))]  # [20160417,20160616)
    test_middle_data = train_data[train_data['date'].isin(pd.date_range('2016/6/16', periods=15))]  # [20160616,20160701)
    test_label_data = test_data.copy()  # [20160701,20160801)


    # train_history_field1 = pd.read_pickle('train_history_field1.pkl')
    # train_label_field1 = pd.read_pickle('train_label_field1.pkl')
    #
    # train_history_field2 = pd.read_pickle('train_history_field2.pkl')
    # train_label_field2 = pd.read_pickle('train_label_field2.pkl')
    #
    # test_history_field = pd.read_pickle('test_history_field.pkl')
    # test_label_field = pd.read_pickle('test_label_field.pkl')
    # online_data = pd.read_pickle('online_data.pkl')
    # online_data.to_pickle('online_data.pkl')
    # train_history_field1.to_pickle('train_history_field1.pkl')
    # train_label_field1.to_pickle('train_label_field1.pkl')
    #
    # train_history_field2.to_pickle('train_history_field2.pkl')
    # train_label_field2.to_pickle('train_label_field2.pkl')
    #
    # test_history_field.to_pickle('test_history_field.pkl')
    # test_label_field.to_pickle('test_label_field.pkl')

    # print('构造训练集')
    # train1 = get_dataset(train_history_field1, online_data, train_label_field1)
    # train2 = get_dataset(train_history_field2, online_data, train_label_field2)
    # print('构造测试集')
    # test = get_dataset(test_history_field, online_data, test_label_field)

    print('构造训练集')
    train1 = get_dataset(train_history_data, online_data, train_label_data)
    print('构造验证集')
    train2 = get_dataset(validate_history_data,online_data, validate_label_data)
    print('构造测试集')
    test = get_dataset(test_history_data, online_data, test_label_data)

    # train1.to_pickle('train1.pkl')
    # train2.to_pickle('train2.pkl')
    # test.to_pickle('test.pkl')
    # train1 = pd.read_pickle('train1.pkl')
    # train2 = pd.read_pickle('train2.pkl')
    # test = pd.read_pickle('test.pkl')
    train = pd.concat([train1, train2], axis=0)
    # #筛特征
    # importance = pd.read_csv('importance.csv', index_col=0)
    # useless = importance[importance['importance'] <= 10]
    # useless.drop('importance', axis = 1, inplace = True)
    # useless = useless.reset_index()
    # useless.drop('index', axis = 1, inplace = True)
    # useless = list(useless['feature_name'])
    # for i in useless :
    #     if i != 'week_6.0':
    #         big_train.drop(i, axis = 1, inplace = True)
    #         test.drop(i, axis = 1, inplace = True)

    result = model_xgb(train, test)
    result = pd.concat([test[['User_id', 'Coupon_id', 'Date_received']], result], axis=1)
    # #0.5 0.5 0.7435
    # #xgb 0.4 lgb 0.2 GBDT 0.4   0.7404
    result.to_csv('out.csv', index = False, header= None)
    # # print('模型特征重要性矩阵')
    # importance.to_csv('importance.csv')
