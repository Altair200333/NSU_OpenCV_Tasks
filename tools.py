import cv2 as cv


def scaleImg(image, scale: float):
    width, height = image.shape[1], image.shape[0]
    return cv.resize(image, (int(width * scale), int(height * scale)))


def clipImg(image, max_size, filter=cv.INTER_CUBIC):
    width, height = image.shape[1], image.shape[0]
    max_dim = max(width, height)
    ratio = float(max_size) / max_dim
    return cv.resize(image, (int(width * ratio), int(height * ratio)), interpolation=filter)


def read_frames(path, frame_size=900):
    cap = cv.VideoCapture(path)
    totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)

    if cap.isOpened() == False:
        print("Error opening video  file")
        return None, -1

    frames = []
    for i in range(int(totalFrames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = clipImg(frame, frame_size)
        frames.append(frame)

    totalFrames = len(frames)

    return frames, totalFrames
