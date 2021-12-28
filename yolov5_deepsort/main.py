import tracker
from detector import Detector
import cv2
import time
from collections import defaultdict
import json


def get_button_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) * 0.5, y2]


def percentage_bar(n, total, time_record=[time.time(), time.time()]):
    """
    :param n int 当前执行任务序列号（从0开始）
    :param total int 总任务数量
    """
    perc = 100 * (n+1) / total
    time_record[-1] = time.time()
    execute_time = time_record[-1] - time_record[0]
    left_time = execute_time * (100 - perc) / perc

    print(
        "{:50s} {:20s} {:<20s} {:<20s}".format(
            "="*int(perc/2),
            "{:.2f}%({}/{})".format(perc, n+1, total),
            "已执行时间(s)：{:.2f}".format(execute_time),
            "预计还需耗时(s)：{:.2f}".format(left_time)),
        end="\r"
        )


def record_bboxs(list_bboxs):
    dict_bboxs = defaultdict(dict)
    for bbox in list_bboxs:
        lbl = ['x1', 'y1', 'x2', 'y2', 'label', 'track_id']
        x1, y1, x2, y2, label, track_id = bbox
        for i in range(len(bbox)):
            dict_bboxs[str(track_id)][lbl[i]] = str(bbox[i])

    with open('frames.csv', 'a') as f:
        f.write(json.dumps(dict_bboxs) + '\n')


if __name__ == '__main__':
    # 初始化 yolov5
    detector = Detector()

    capture = cv2.VideoCapture('/Users/zhouchenye/Desktop/PythonProject/video_system_data/00028/00028.mp4')
    cnt_frames, cnt = capture.get(7), 0
    _, im = capture.read()
    while im is not None:
        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))
        bboxes = detector.detect(im)

        if bboxes:
            list_bboxs = tracker.update(bboxes, im)
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=2)
        else:
            output_image_frame = im

        if list_bboxs:
            record_bboxs(list_bboxs)

        # cv2.imshow('demo', output_image_frame)
        # cv2.waitKey(1)
        percentage_bar(cnt, cnt_frames)
        cnt += 1

        _, im = capture.read()

    capture.release()
    cv2.destroyAllWindows()
