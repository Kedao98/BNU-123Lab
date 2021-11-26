import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def calculate_transform_matrix():
    """
    位置点 X Y
    左上 513 522
    右上 1044 521
    左下 476 643
    右下 1169 635

    """
    pts_o = np.float32([[513, 522], [1044, 521], [476, 643], [1169, 635]])
    pts_d = np.float32([[0, 0], [100, 0], [0, 100], [100, 100]])
    return cv2.getPerspectiveTransform(pts_o, pts_d)


def calculate_button_point(data):
    row = data.split(':')[2:]
    row = [''.join(list(filter(lambda x: x in '0123456789', r))) for r in row]
    x = float(row[0]) + float(row[2])/2
    y = float(row[1]) + float(row[3])
    return [x, y]


def parse_frame(file):
    with open(file, 'r')as f:
        frame = f.read().split('\n')
    return [calculate_button_point(item) for item in frame if item.split(':')[0] == 'person']
    # res = []
    # for item in frame:
    #     if item.split(':')[0] != 'person':
    #         continue
    #     res.append(calculate_button_point(item))
    # return res


def perspective_transform(frame, M):
    res = []
    for object in frame:
        temp = M * np.mat([object[0], object[1], 1]).T
        temp = [float(i) for i in temp]
        res.append([temp[0]/temp[2], temp[1]/temp[2]])
    return res


def draw_pic(frame, save_file):
    plt.figure()
    plt.scatter([item[0] for item in frame], [item[1] for item in frame], s=50)
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.savefig(save_file)
    plt.close()


def main(source_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    M = calculate_transform_matrix()
    for i in range(len(os.listdir(source_path))):
        full_name = '{}/{}.txt'.format(source_path, i)
        frame = parse_frame(full_name)
        frame = perspective_transform(frame, M)
        draw_pic(frame, '{}/{}.jpg'.format(save_path, i))


if __name__ == '__main__':
    source_path = './frame_split'
    save_path = './transform_res'
    main(source_path, save_path)
