import pandas as pd
import numpy as np
import os


class Markov:
    def __init__(self, state_list, absorbing_state, source_file):
        """
        :param state_list: list 状态列表
        :param absorbing_state: list 吸收态列表 将tc置于列表末尾
        :param source_file:
        """
        self.absorbing_state = absorbing_state
        self.state_list = state_list + absorbing_state
        # self.origin_matrix = self.passchain2matrix(self.load_passchain(source_file), self.state_list)
        self.origin_matrix = self.load_data(source_file)
        self.pass_prob_matrix = self.calc_prob_matrix(self.origin_matrix)
        self.tc0 = self.predict(self.pass_prob_matrix)
        self.res_list = None

    def load_data(self, source_file):
        """
        :param source_file: 数据来源 文件/文件夹
        :return:np.mat from-to计数矩阵
        """
        def load_passchain(source_file):
            if '.' in os.path.split(source_file)[-1]:  # 某个文件
                return pd.read_excel(source_file, dtype=str)

            pass_chains = pd.DataFrame()
            for file in os.listdir(source_file):
                if file == '.DS_Store':
                    continue
                pass_chains = pass_chains.append(pd.read_excel(os.path.join(source_file, file), dtype=str))
            return pass_chains

        pass_chain = load_passchain(source_file)
        pass_table = [[0] * len(self.state_list) for _ in range(len(self.state_list))]
        for i, row in pass_chain.iterrows():
            if pd.isna(row['PLAYER']):
                continue
            pass_table[self.state_list.index(row['PLAYER'])][self.state_list.index(row['NEXT PLAYER'])] += 1
        return np.mat(pass_table)

    @staticmethod
    def calc_prob_matrix(cnt_matrix):
        cnt_matrix[-1, -1], cnt_matrix[-2, -2] = 1, 1
        return cnt_matrix / cnt_matrix.sum(axis=1)

    def predict(self, matrix, init_state=None, max_iter=100, threshold=1e-10):
        """
        :param matrix: np.mat 概率矩阵
        :param init_state:np.mat 初始状态
        :param max_iter:int 最大迭代次数
        :param threshold:float 误差终止条件
        :return:int 终态TC概率
        """
        def distance(vec1, vec2):
            return float(sum((vec1 - vec2) * (vec1 - vec2).T))

        if not init_state:
            init_state = self.origin_matrix.sum(axis=0)
            init_state[0, -len(self.absorbing_state):] = 0
            init_state = init_state / init_state.sum()
            # init_state = np.mat([1/matrix.shape[1] for _ in range(matrix.shape[1])])

        matrix = matrix[:]
        for _ in range(max_iter):
            state = init_state * matrix
            if distance(init_state, state) < threshold:
                break
            init_state = state
        return state[0, -1]

    @staticmethod
    def deflect(matrix, pos, c=1, b=5):
        """
        :param matrix: np.mat 原概率矩阵
        :param pos: (i, j) 偏转位置
        :param c: int
        :param b: int
        :return: np.mat 偏转后的矩阵
        """
        matrix, (i, j) = matrix[:], pos
        delta = (c + b * 4 * matrix[i, j]) / 100
        for ele in range(matrix[i].shape[1]):
            if ele == j:
                continue
            matrix[i, ele] -= delta * matrix[i, ele] / (1 - matrix[i, j])
        matrix[i, j] += delta
        return matrix

    def get_deflected_result(self):
        """
        :return:np.array 该矩阵各联系偏转后TC变化量
        """
        shape = self.origin_matrix.shape
        self.res_list = np.zeros(shape)
        for i in range(shape[0] - len(self.absorbing_state)):
            for j in range(shape[0] - len(self.absorbing_state)):
                deflected_mat = self.deflect(self.pass_prob_matrix[:], (i, j))
                self.res_list[i][j] = (self.predict(deflected_mat) - self.tc0) * 100
        return self.res_list

    def save_result(self, save_path):
        if self.res_list is None:
            print('请先运行get_deflected_result')
        else:
            save_df = pd.DataFrame(self.res_list)
            save_df.columns = save_df.index = self.state_list
            save_df.to_csv(save_path)


if __name__ == '__main__':
    source_file = './data/shenhuaVSjiangsu.xls'
    # state_list = ['RDM', 'ROM', 'RA', 'ZOM', 'LDM', 'ZS', 'RS', 'ZDM',
    #               'LOM', 'LS', 'ZA', 'LA', 'NTC', 'TC']
    absorbing_state = ['NTC', 'TC']
    state_list = [i for i in "1 2 3 5 7 10 13 20 22 23 26".split(' ')]
    model = Markov(state_list, absorbing_state, source_file)
    model.get_deflected_result()
    model.save_result('shenhuaVSjiangsu.csv')
