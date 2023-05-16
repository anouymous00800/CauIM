import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from HyperSpreading import Hyperspreading, Hyperspreading_ITE
from tqdm import tqdm
import copy
import random
from DataTransform import *
import networkx as nx
import matplotlib
import os
import time
matplotlib.use('Agg')
plt.switch_backend('agg')


def generalCELF(df_hyper_matrix, K, R, hs, pic_path):
    """
    GeneralCELF algorithm
    """
    degree = df_hyper_matrix.sum(axis=1) #对横坐标每个节点求超边数之和，为degree
    #degree.sort_values(ascending=False)
    degree = degree[degree>0] #只看degree>0的点
    inf_spread_matrix = []
    start_time = time.time()

    node_list = []
    for node,v in degree.items():
        node_list.append(node)

    marg_gain = [hs.hyperCELFSI(R, df_hyper_matrix, [node]) for node in node_list]  # 加进去
 
    # Create the sorted list of nodes and their marginal gain
    # 从大到小按照后面的值排序，是个元组列表
    Q = sorted(zip(node_list, marg_gain), key=lambda x: x[1], reverse=True) 
    
 
    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [len(node_list)], [time.time() - start_time]
 
    for _ in range(K - 1):
 
        check, node_lookup = False, 0
 
        while not check:
            # Count the number of times the spread is computed
            node_lookup += 1
 
            # Recalculate spread of top node
            current = Q[0][0]
 
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, hs.hyperCELFSI(R, df_hyper_matrix, S+[current]) - spread)
 
            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
 
            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)
 
        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)#每个种子影响力
        LOOKUPS.append(node_lookup) # 查找数目统计
        timelapse.append(time.time() - start_time)
 
        # Remove the selected node from the list
        Q = Q[1:]
    '''
    hs_test = Hyperspreading_ITE()
    ite_list =[]
    for i in range(0,K):
        scale = hs_test.hyperCELF_ITE(R, df_hyper_matrix, S[0:i],ite)
        ite_list.append(scale)
        
    '''
    return (S, SPREAD, timelapse, LOOKUPS)



def generalCausalCELF(df_hyper_matrix, ite, K, R, hs_ite, pic_path, p):
    """
    CausalCELF algorithm
    """
    degree = df_hyper_matrix.sum(axis=1) #对每个节点求超边数，为degree
    #degree.sort_values(ascending=False)
    degree = degree[degree>0] #只看degree>0的点
    inf_spread_matrix = []
    start_time = time.time()

    node_list = []
    for node,v in degree.items():
        node_list.append(node)

    marg_gain = [hs_ite.hyperCELF_ITE(R, df_hyper_matrix, [node],ite, p) for node in node_list]  # 加进去
 
    # Create the sorted list of nodes and their marginal gain
    # 从大到小按照后面的值排序，是个元组列表
    Q = sorted(zip(node_list, marg_gain), key=lambda x: x[1], reverse=True) 
    
 
    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [len(node_list)], [time.time() - start_time]
    count = 0 #计数种子
    print('in celf, K',K)
    for _ in range(K - 1):
        count += 1
        print('seed: ', count)
 
        check, node_lookup = False, 0
 
        while not check:
            # Count the number of times the spread is computed
            node_lookup += 1
 
            # Recalculate spread of top node
            current = Q[0][0]
 
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, hs_ite.hyperCELF_ITE(R, df_hyper_matrix, S+[current], ite, p) - spread)
 
            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
 
            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)
 
        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)#每个种子加入后影响力（ITE之和）
        LOOKUPS.append(node_lookup) # 查找数目统计
        timelapse.append(time.time() - start_time)
 
        # Remove the selected node from the list
        Q = Q[1:]
    final_scale_list_mean = SPREAD
    seed_ite_info = {}
    for v in S:
        seed_ite_info[v] = ite[v]
    # return (S, SPREAD, timelapse, LOOKUPS)
    return timelapse, final_scale_list_mean, seed_ite_info


def IM_EXP(n = 15, R = 10):
    hs = Hyperspreading() #传统IM
    re = 18 # reviewer
    book_min = 10 # book_num_min
    book_max = 15
    type_s = 'linear'  # linear, quadratic
    nonlinear_type = 'raw'
    alpha = 1.0
    beta = 1.0
    
    pic_path = 'pic_'+str(re)+'_' +str(book_min) +'_'+ str(book_max)
    pp = 'filter_'+str(re)+'_' +str(book_min) +'_'+ str(book_max)+'_'
    path_save = pp +'GoodReads_sim_' + type_s + '_alpha' + str(alpha) + '_beta' + str(beta) +\
                '_nonlinear_' + nonlinear_type 
    filename = './data/pic/'+pic_path
    if not os.path.exists(filename):
        message = "Make new figure file."
        print(message)
        os.mkdir(filename)

    df_hyper_matrix, N, ite = changeEdgeToMatrix2('./data/Simulation/GR/' + path_save + '.mat')

  # 25, 50
    num_seed = n
    Rounds = R
    
    #(S, SPREAD, timelapse, LOOKUPS) = generalCELF(df_hyper_matrix, num_seed, Rounds, hs, pic_path)
    res = generalCELF(df_hyper_matrix, num_seed, Rounds, hs, pic_path)
    #ggd_scale_list, seed_ite_info = generalCausalCELF(df_hyper_matrix, ite, num_seed, Rounds)
   
    #存储 每个种子加入后（即不同种子数目）影响力
    final_matrix = []
    final_matrix.append(res[1])

    final_df = pd.DataFrame(final_matrix).T
    final_df.columns = [['CELF']]
    # print(final_df)
    final_df.to_csv('./data/result/CELF/IM/' + path_save + '.csv')


def IM_CAUSAL(n = 15, R = 10, p=0.01):
    # print("beta: ",beta)
    hs_ite = Hyperspreading_ITE()   # ITE 方法
    pic_path = ''
    ''' ds: goodreads
    re = 18 # reviewer
    book_min = 10 # book_num_min
    book_max = 15

    type_s = 'linear'  # linear, quadratic
    nonlinear_type = 'raw'
    alpha = 1.0
    beta = 1.0
    

    pic_path = 'causal_pic_'+str(re)+'_' +str(book_min) +'_'+ str(book_max)
    pp = 'filter_'+str(re)+'_' +str(book_min) +'_'+ str(book_max)+'_'
    path_save = pp +'GoodReads_sim_' + type_s + '_alpha' + str(alpha) + '_beta' + str(beta) +\
                '_nonlinear_' + nonlinear_type 
    filename = './data/pic/'+pic_path
    if not os.path.exists(filename):
        message = "Make new figure file."
        print(message)
        os.mkdir(filename)
    '''
    # ds: contact
    type = 'linear'  # linear, quadratic
    nonlinear_type = 'raw'
    alpha = 1.0
    beta = 1.0
    nn = 327#节点
    mm = 2320#超边
    pp = 'filter_n_'+str(nn)+'_m_' +str(mm) +'_'
    path_save = pp +'contact_sim_' + type + '_alpha' + str(alpha) + '_beta' + str(
         beta) + '_nonlinear_' + nonlinear_type
        

    df_hyper_matrix, N, ite = changeEdgeToMatrix2('./data/Simulation/contact/' + path_save + '.mat')

# 25, 50
    num_seed = n
    Rounds = R
    
    #算法模块
    #ggd_scale_list = generalCELF(df_hyper_matrix, num_seed, Rounds)
    timelapse, ggd_scale_list_mean, seed_ite_info = generalCausalCELF(df_hyper_matrix, ite, num_seed, Rounds, hs_ite, pic_path, p)
    

    print("gdd_shape:",len(ggd_scale_list_mean))
    #存储 ITE 即为影响力

    print("Time used:", timelapse)

    return timelapse, ggd_scale_list_mean,path_save



if __name__ == '__main__':
    num_seed = 15
    Rounds = 20
    T= 10
    beta_list = [0.001,  0.003,  0.005, 0.007, 0.009, 0.012, 0.015]
    beta_list2 = [0.01]

    num = 1
    for p in beta_list2:
        print("p:", p)
        final_df = []
        time_df = []
        for i in range(0, T):
            print('test: ',i)
            timelapse, ggd_scale_list_mean, path_save = IM_CAUSAL(num_seed, Rounds, p)
            print(ggd_scale_list_mean)
            final_df.append(ggd_scale_list_mean)
            time_df.append(timelapse)

        final_list = pd.DataFrame(final_df).T
        #final_list.columns = [['T1','T2','T3']]
        #final_list.columns = [['T1','T2','T3','T4','T5']]
        final_list.columns = [['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10']]
        #final_list.to_csv('/Users/suxinyan/Desktop/result/p/ITE_p_x'+ str(num)+'.csv')
        final_list.to_csv('/Users/suxinyan/Desktop/CauIM/result/contact/ITE_celfcauim_iter'+ str(25)+'.csv')

        time_list = pd.DataFrame(time_df).T
        #time_list.columns = [['T1','T2','T3']]
        time_list.columns = [['T1','T2','T3','T4','T5','T6','T7','T8','T9','T10']]
        #time_list.columns = [['T1','T2','T3','T4','T5']]
        #time_list.to_csv('/Users/suxinyan/Desktop/result/p/Time_p_x'+ str(num)+'.csv')
        time_list.to_csv('/Users/suxinyan/Desktop/CauIM/result/contact/Time_celfcauim_iter'+ str(25)+'.csv')
        num += 1
        #IM_EXP(num_seed,Rounds)



   