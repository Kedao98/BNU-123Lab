import cv2


def fetch_point(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


def video_flag(video):
    frame = 0
    cap = cv2.VideoCapture(video)

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("video", fetch_point, frame)

    while True:
        ret, frame = cap.read()
        cv2.imshow("video", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


video_flag("../video_system_data/00006/00006_res.mp4")
