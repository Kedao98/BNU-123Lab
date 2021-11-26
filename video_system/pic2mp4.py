import cv2
import os

img_root = './transform_res'  # 是图片序列的位置
fps = 12  # 可以随意调整视频的帧速率

#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('TestVideo_1.avi', fourcc, fps, (640, 480))

for i in range(len(os.listdir(img_root))):
    if i % 1000 == 0:
        print(i)
    frame = cv2.imread('{}/{}.jpg'.format(img_root, i))
    # cv2.imshow('frame', frame)
    videoWriter.write(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
videoWriter.release()
cv2.destroyAllWindows()
