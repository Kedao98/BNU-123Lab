import pandas as pd
import os


def passchain2matrix(df, title):
    pass_table = [[0] * len(title) for _ in range(len(title))]
    for i, row in df.iterrows():
        if pd.isnull(row['ZONE']):
            continue
        pass_table[title.index(row['ZONE'])][title.index(row['NEXT ZONE'])] += 1
    return pass_table


def main(source_path, save_path, title):
    for file in os.listdir(source_path):
        pass_chain = pd.read_excel(os.path.join(source_path, file))
        pass_matrix = passchain2matrix(pass_chain, title)


if __name__ == '__main__':
    title = ['RDM', 'ROM', 'RA', 'ZOM', 'LDM', 'ZS', 'RS', 'ZDM',
             'LOM', 'LS', 'ZA', 'LA', 'NTC', 'TC']
    source_path = './data'
    save_path = ''
    main(source_path, save_path, title)
