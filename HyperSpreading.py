import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Hyperspreading:

    def getHpe(inode, matrix):

        return np.where(matrix[inode, :] == 1)[0]

    def chooseHpe(hpe_set):

        if len(hpe_set) > 0:
            return random.sample(list(hpe_set), 1)[0]
        else:
            return []

    def getNodesofHpe(hpe, matrix):

        return np.where(matrix[:, hpe] == 1)[0]

    def getNodesofHpeSet(hpe_set, matrix):

        adj_nodes = []
        for hpe in hpe_set:
            adj_nodes.extend(Hyperspreading.getNodesofHpe(hpe, matrix))
        return np.array(adj_nodes)

    def findAdjNode_CP(inode, df_hyper_matrix):

        edges_set = Hyperspreading.getHpe(inode, df_hyper_matrix.values)

        edge = Hyperspreading.chooseHpe(edges_set)

        adj_nodes = np.array(Hyperspreading.getNodesofHpe(edge, df_hyper_matrix.values))
        return adj_nodes

    def formatInfectedList(I_list, infected_list, infected_T):

        return (x for x in infected_list if x not in I_list and x not in infected_T)

    def getTrueStateNode(self, adj_nodes, I_list, R_list):

        adj_list = list(adj_nodes)
        for i in range(0, len(adj_nodes)):
            if adj_nodes[i] in I_list or adj_nodes[i] in R_list:
                adj_list.remove(adj_nodes[i])
        return np.array(adj_list)

    def spreadAdj(adj_nodes, I_list, infected_T, beta):

        random_list = np.random.random(size=len(adj_nodes))
        infected_list = adj_nodes[np.where(random_list < beta)[0]]
        infected_list_unique = Hyperspreading.formatInfectedList(I_list, infected_list, infected_T)
        return infected_list_unique

    # 复杂版的传播函数，返回更多信息
    def hyperSI(self, draw, r, df_hyper_matrix, seeds, pic_path, ite):

        I_list = list(seeds)
        beta = 0.01
        iters = 25
        I_total_list = [1]

        sum_seed = 0
        for v in seeds:
            sum_seed += ite[v] #计算种子ITE之和

        for t in range(0, iters):
            infected_T = []
            for inode in I_list:
                adj_nodes = Hyperspreading.findAdjNode_CP(inode, df_hyper_matrix)
                infected_list_unique = Hyperspreading.spreadAdj(adj_nodes, I_list, infected_T, beta)
                infected_T.extend(infected_list_unique)
            I_list.extend(infected_T)
            I_total_list.append(len(I_list))
        '''
        if draw == True:
            plt.title("r_"+str(r))
            plt.plot(np.arange(np.array(len(I_total_list))),I_total_list,color='orange')
            # plt.plot(x, y, format_string, **kwargs) 
            plt.savefig('./data/pic/'+ pic_path+ '/r_'+str(r)+'result.png')
            plt.pause(1)
            plt.close()
        '''
        sum_all = 0
        for x in  I_list:
            sum_all += ite[x]
        ITE_total = sum_all - sum_seed

        return I_total_list[-1:][0]-len(list(seeds)), I_total_list, I_list, ITE_total# 整体影响(在列表最后一项记录,除去一开始的seed数目)，每个时间步影响力之和(列表形式，对应不同时间步)，影响的全部具体节点


    def hyperCELFSI(self, R, df_hyper_matrix, seeds):      
        beta = 0.01
        iters = 25

        for r in range(0, R):
            #print("Round: ",r)
            #I_total_list = [1]
            sum_all = []
            I_list = list(seeds)
            for t in range(0, iters):
                #if t == iters-1:
                #   print("Iter: ",iters,"!")
                infected_T = []
                for inode in I_list:
                    adj_nodes = Hyperspreading.findAdjNode_CP(inode, df_hyper_matrix)
                    infected_list_unique = Hyperspreading.spreadAdj(adj_nodes, I_list, infected_T, beta)
                    infected_T.extend(infected_list_unique)
                I_list.extend(infected_T) #拓展I_list
                #I_total_list.append(len(I_list))#演变过程
            sum_all.append(len(I_list)-len(list(seeds)))#计算25个时间步后影响的个体总数,不算seed
        
        sum_real = np.mean(sum_all) #多轮(R)求平均
        #return sum_all,I_total_list, I_list # 整体影响，每个时间步影响力之和(列表形式，对应不同时间步)，影响的全部节点
        return sum_real


class Hyperspreading_ITE:

    def getHpe(inode, matrix):

        return np.where(matrix[inode, :] == 1)[0]

    def chooseHpe(hpe_set):

        if len(hpe_set) > 0:
            return random.sample(list(hpe_set), 1)[0]
        else:
            return []

    def getNodesofHpe(hpe, matrix):

        return np.where(matrix[:, hpe] == 1)[0]

    def getNodesofHpeSet(hpe_set, matrix):

        adj_nodes = []
        for hpe in hpe_set:
            adj_nodes.extend(Hyperspreading_ITE.getNodesofHpe(hpe, matrix))
        return np.array(adj_nodes)

    def findAdjNode_CP(inode, df_hyper_matrix):

        edges_set = Hyperspreading_ITE.getHpe(inode, df_hyper_matrix.values)# 包含该节点的超边集合，返回是np.array

        edge = Hyperspreading_ITE.chooseHpe(edges_set)# 随机只选择一条超边

        adj_nodes = np.array(Hyperspreading_ITE.getNodesofHpe(edge, df_hyper_matrix.values))# 该超边包含的所有节点
        return adj_nodes

    def formatInfectedList(I_list, infected_list, infected_T):

        return (x for x in infected_list if x not in I_list and x not in infected_T)

    def getTrueStateNode(self, adj_nodes, I_list, R_list):

        adj_list = list(adj_nodes)
        for i in range(0, len(adj_nodes)):
            if adj_nodes[i] in I_list or adj_nodes[i] in R_list:
                adj_list.remove(adj_nodes[i])
        return np.array(adj_list)

    def spreadAdj(adj_nodes, I_list, infected_T, beta):

        random_list = np.random.random(size=len(adj_nodes))#随机出这些节点的数值，区间在[0,0.01]才被传上
        infected_list = adj_nodes[np.where(random_list < beta)[0]]
        infected_list_unique = Hyperspreading_ITE.formatInfectedList(I_list, infected_list, infected_T)
        return infected_list_unique
    # 画图版
    def hyperSI_ITE(self, draw, r, df_hyper_matrix, ite, seeds, pic_path):
        I_list = list(seeds)
        beta = 0.01
        iters = 25
        ITE_total_list = [0]
        # sum_all = 0 # 记录所有的时间步内的ITE之和
        sum_seed = 0
        for v in seeds:
            sum_seed += ite[v] #计算种子ITE之和

        for t in range(0, iters):
            infected_T = []
            for inode in I_list:
                adj_nodes = Hyperspreading_ITE.findAdjNode_CP(inode, df_hyper_matrix)#邻近节点
                infected_list_unique = Hyperspreading_ITE.spreadAdj(adj_nodes, I_list, infected_T, beta)#i_list是seed
                infected_T.extend(infected_list_unique)
            I_list.extend(infected_T)
            # I_total_list改成ite
            sum0 = 0
            for x in  I_list:
                sum0 += ite[x]
            ITE_total_list.append(sum0) # 记录变化
        if draw == True:
            plt.title("r_"+str(r))
            plt.xlabel("Step T ")
            plt.ylabel("Sum_of_ITE")
            plt.plot(np.arange(np.array(len(ITE_total_list))),ITE_total_list,color='orange')
            # plt.plot(x, y, format_string, **kwargs) 
            plt.savefig('./data/pic/'+ pic_path +'/r_'+str(r)+'result.png')
            plt.close()
        return ITE_total_list, ITE_total_list[-1:][0]-sum_seed, I_list #参数2表示一共25个时间步，返回存储的ITE总和，和影响的节点列表（目前不用）

    #简约版
    def hyperSI_ITE_simple(self, r, df_hyper_matrix, ite, seeds):
        I_list = list(seeds)
        beta = 0.01
        iters = 25
        sum_all = 0 # 记录所有的时间步内的ITE之和
        sum_seed = 0
        for v in seeds:
            sum_seed += ite[v] #计算种子ITE之和

        for t in range(0, iters):
            infected_T = []
            for inode in I_list:
                adj_nodes = Hyperspreading_ITE.findAdjNode_CP(inode, df_hyper_matrix)#邻近节点
                infected_list_unique = Hyperspreading_ITE.spreadAdj(adj_nodes, I_list, infected_T, beta)#i_list是seed
                infected_T.extend(infected_list_unique)
            I_list.extend(infected_T)
            # I_total_list改成ite

        for x in  I_list:
            sum_all += ite[x]

        return sum_all - sum_seed #ite之和


    def hyperCELF_ITE(self, R, df_hyper_matrix, seeds, ite, p):      
        beta = p
        iters = 25
        sum_all = []#记录每一轮的ITE
        sum_seed = 0
        for v in seeds:
            sum_seed += ite[v] #计算种子ITE之和

        #for r in tqdm(range(R), desc="Loading..."):
        for r in range(0, R):
            #print("Round: ",r)
            #I_total_list = [1]
            
            I_list = list(seeds)
            for t in range(0, iters):
                #if t == iters-1:
                #    print("Iter: ",iters,"!")
                infected_T = []
                for inode in I_list:
                    adj_nodes = Hyperspreading.findAdjNode_CP(inode, df_hyper_matrix)
                    infected_list_unique = Hyperspreading.spreadAdj(adj_nodes, I_list, infected_T, beta)
                    infected_T.extend(infected_list_unique)
                I_list.extend(infected_T) #拓展I_list
                #I_total_list.append(len(I_list))#演变过程
            #一轮结束，开始计算ITE之和
            temp = 0
            for x in  I_list:
                temp += ite[x]
            sum_all.append(temp-sum_seed)#计算25个时间步后影响的个体ITE之和，去掉seed的
        
        sum_real = np.mean(sum_all) #多轮(R)求平均
        #return sum_all,I_total_list, I_list # 整体影响，每个时间步影响力之和(列表形式，对应不同时间步)，影响的全部节点
        return sum_real

     

     