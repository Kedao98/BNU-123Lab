import pandas as pd
import numpy as np
import os


class Markov:

    def __init__(self, state_list, source_file):
        self.state_list = state_list
        self.origin_matrix = self.passchain2matrix(self.load_passchain(source_file), state_list)
        self.pass_prob_matrix = self.calc_prob_matrix(self.origin_matrix)
        self.tc0 = self.predict(self.pass_prob_matrix)
        self.res_list = None

    def get_deflected_result(self):
        """
        :return:np.array 该矩阵各联系偏转后TC变化量
        """
        shape = self.origin_matrix.shape
        self.res_list = np.zeros(shape)
        for i in range(shape[0] - 1):
            for j in range(i, shape[0]):
                self.res_list[i][j] = (self.predict(self.deflect(self.origin_matrix[:], (i, j))) - self.tc0) * 100
        return self.res_list

    def load_passchain(self, source_file):
        if '.' in os.path.split(source_file)[-1]:# 某个文件
            return pd.read_excel(source_file)

        pass_chains = pd.DataFrame()
        for file in os.listdir(source_file):
            if file == '.DS_Store':
                continue
            pass_chains = pass_chains.append(pd.read_excel(os.path.join(source_file, file)))
        return pass_chains


    def passchain2matrix(self, pass_chain, state_list):
        """
        :param pass_chain: pd.DataFrame 原始采集数据
        :param state_list: list 状态列表，将tc置于列表末尾
        :return:np.mat from-to计数矩阵
        """
        pass_table = [[0] * len(state_list) for _ in range(len(state_list))]
        for i, row in pass_chain.iterrows():
            if pd.isna(row['ZONE']):
                continue
            pass_table[state_list.index(row['ZONE'])][state_list.index(row['NEXT ZONE'])] += 1
        return np.mat(pass_table)

    def calc_prob_matrix(self, cnt_matrix):
        cnt_matrix[-1, -1], cnt_matrix[-2, -2] = 1, 1
        return cnt_matrix / cnt_matrix.sum(axis=1)

    def predict(self, matrix, init_state=None, max_iter=100, threshold=1e-5):
        """
        :param matrix: np.mat 概率矩阵
        :param init_state:np.mat 初始状态
        :param max_iter:int 最大迭代次数
        :param threshold:float 误差终止条件
        :return:int 终态TC概率
        """
        if not init_state:
            init_state = np.mat([1/matrix.shape[1] for _ in range(matrix.shape[1])])
        for _ in range(max_iter):
            state = init_state * matrix
            if self.distance(init_state, state) < threshold:
                break
            init_state = state
        return state[-1]

    def deflect(self, matrix, pos, c=1, b=5):
        """
        :param matrix: np.mat 原概率矩阵
        :param pos: (i, j) 偏转位置
        :param c: int
        :param b: int
        :return: np.mat 偏转后的矩阵
        """
        i, j = pos
        delta = (c + b * 4 * matrix[i, j]) / 100
        for ele in range(matrix[i].shape[1]):
            if ele == j:
                matrix[i, ele] += delta
            else:
                matrix[i, ele] -= delta * matrix[i, ele] / (1 - matrix[i, j])
        return matrix

    def distance(self, vec1, vec2):
        return sum((vec1 - vec2) * (vec1 - vec2).T)[0, 0]

    def save_result(self, save_path):
        if not self.res_list:
            print('请先运行get_deflected_result')
        else:
            save_df = pd.DataFrame(self.res_list)
            save_df.columns = save_df.index = self.state_list
            save_df.to_csv(save_path)


if __name__ == '__main__':
    source_file = './data'
    state_list = ['RDM', 'ROM', 'RA', 'ZOM', 'LDM', 'ZS', 'RS', 'ZDM',
                  'LOM', 'LS', 'ZA', 'LA', 'NTC', 'TC']
    model = Markov(state_list, source_file)
    model.get_deflected_result()
