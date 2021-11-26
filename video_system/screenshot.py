import cv2
import os


class Screenshot(object):
    def __init__(self, video, frame_info_source, save_path):
        self.video = video
        self.save_path = save_path
        self.frame_items_source = frame_info_source
        self.cnt = 0

    def process(self):
        video = cv2.VideoCapture(self.video)
        video_not_end, frame = video.read()
        box_list = self.get_box_list()
        while video_not_end:
            item_img_list = [self.get_item_screenshot(frame, box) for box in box_list.__next__()]
            self.save_img(item_img_list)
            video_not_end, frame = video.read()
        video.release()

    def save_img(self, img_list):
        os.makedirs(self.save_path, exist_ok=True)
        for img in img_list:
            cv2.imwrite(os.path.join(self.save_path, '{}.jpg'.format(self.cnt)), img)
            self.cnt += 1

    def parse_frame_items(self, file):
        """
        :param file:某帧目标检测结果txt文件名
        :return: [[x, y, w, h],...]
        """
        content = []
        for line in open(file, 'r'):
            line = line.replace('\n', '').split('\t')[-1].strip('()').split(' ')
            content.append([int(item) for item in line if item and ':' not in item])
        return content

    def get_box_list(self):
        for i in range(len(os.listdir(self.frame_items_source))):
            yield self.parse_frame_items(os.path.join(self.frame_items_source, '{}.txt'.format(i)))

        # file_list = [file for file in os.listdir(self.frame_items_source) if file != '.DS_Store']
        # for file in sorted(file_list, key=lambda x: int(x.split('.txt')[0])):
        #     yield self.parse_frame_items(os.path.join(self.frame_items_source, file))

    def get_item_screenshot(self, frame, box):
        """
        :param frame: 视频单帧截图
        :param box: 边界框
        :return: 单个对象截图
        """
        x, y, w, h = box
        return frame[y:y + h, x:x + w]


if __name__ == '__main__':
    video = '../video_system_data/00006/00006.MP4'
    save_path = './screenshot'
    frame_info_source = './frame_split'
    ss = Screenshot(video, frame_info_source, save_path)
    ss.process()
