import collections
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import base64
import io
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def cal_cross(points):
    vec1 = np.array(points[1]) - np.array(points[0])
    vec2 = np.array(points[2]) - np.array(points[0])
    return np.cross(vec1, vec2)


def rgb2lab(rgb):
    r, g, b = rgb / 255.0

    # gamma 2.2
    r, g, b = [pow((i + 0.055) / 1.055, 2.4) if i > 0.04045 else i / 12.92 for i in [r, g, b]]

    # sRGB
    X = (r * 0.436052025 + g * 0.385081593 + b * 0.143087414) * 100 / 96.4221
    Y = (r * 0.222491598 + g * 0.716886060 + b * 0.060621486) * 100 / 100.000
    Z = (r * 0.013929122 + g * 0.097097002 + b * 0.714185470) * 100 / 82.5211

    # Lab
    X, Y, Z = [pow(i, 1 / 3.000) if i > 0.008856 else (7.787 * i) + (16 / 116.000) for i in [X, Y, Z]]

    Lab_L = round((116.000 * Y) - 16.000, 2)
    Lab_a = round(500.000 * (X - Y), 2)
    Lab_b = round(200.000 * (Y - Z), 2)

    return [Lab_L, Lab_a, Lab_b]


def pixel_cluster(data):
    kmeans = KMeans(n_clusters=10).fit(data)
    cnt = collections.defaultdict(int)
    for label in kmeans.labels_:
        cnt[label] += 1
    return [kmeans.cluster_centers_[i] for i, _ in sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)[:3]]


def get_dominant_colors_kmeans(infile):
    image = Image.open(infile)
    small_image = image.resize((80, 80)) if image.size[0] * image.size[1] > 80 * 80 else image
    flatten_lab_image = [rgb2lab(i) for i in np.array(small_image).reshape(-1, 3)]
    return pixel_cluster(flatten_lab_image)


def get_dominant_colors(infile):
    image = Image.open(infile)

    # 缩小图片，否则计算机压力太大
    small_image = image.resize((80, 80)) if image.size[0] * image.size[1] > 80 * 80 else image
    # 找到主要的颜色
    result = small_image.convert("P", palette=Image.ADAPTIVE, colors=10)

    palette = result.getpalette()
    color_counts = sorted(result.getcolors(), reverse=True)
    colors = list()

    for i in range(3):    # 3个主要颜色的图像
        palette_index = color_counts[i][1]
        dominant_color = palette[palette_index * 3: palette_index * 3 + 3]
        colors.append(tuple(dominant_color))
    return cal_cross(colors)


def get_rgb_hist_feature(image_name):
    hist_feature = []
    img = np.array(Image.open(image_name))
    color = ('b', 'g', 'r')
    for index, value in enumerate(color):
        hist = cv2.calcHist([img], [index], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist_feature.append(hist)
    return list(np.array(hist_feature).ravel())


def calculate_dis(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return abs(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))


def load_json(json_file):
    with open(json_file, 'r')as f:
        return json.loads(f.read())


def parse_base64(base):
    return io.BytesIO(base64.b64decode(base))


def lr(data):
    data = pd.DataFrame(data)
    exam_X, exam_Y = data.iloc[:, :-1], data.iloc[:, -1]
    # X_train, X_test, Y_train, Y_test = train_test_split(exam_X.values.reshape(-1, 1), exam_Y.values.reshape(-1, 1), train_size=.8)
    X_train, X_test, Y_train, Y_test = train_test_split(exam_X.values, exam_Y.values, train_size=.8)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    print(model.score(X_test, Y_test))


def kmeans(data):
    data = pd.DataFrame(data)
    exam_X, exam_Y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(exam_X, exam_Y, train_size=.8)
    model = KMeans(n_clusters=2, random_state=0).fit(X_train)
    print(model.score(X_test))


def temp(source_file):
    data = [load_json(os.path.join(source_file, file)) for file in os.listdir(source_file)
            if file.split('.')[-1] == 'json']
    # img_vec_list = [cal_cross(get_dominant_colors_kmeans(parse_base64(i['imageData']))) for i in data]
    img_vec_list = [get_dominant_colors_kmeans(parse_base64(i['imageData'])) for i in data]
    # img_list = [get_rgb_hist_feature(parse_base64(i['imageData'])) for i in data]
    dis_list = [[calculate_dis(img_vec_list[i], img_vec_list[j]), data[i]['flags'] == data[j]['flags']]
                for i in range(len(data)) for j in range(i, len(data))]
    lr(dis_list)


if __name__ == '__main__':
    source_file = './screenshot_annotation'
    temp(source_file)
