from S3FDExecute import *

def detect_face(extractor, img_path):
    input_image = cv2.imread(img_path)
    if input_image is None:
        logging.error(f"Load {img_path} Failed")
        exit(1)
    detected_faces = extractor.extract(input_image,
                                       is_resize=False,
                                       min_pixel_threshold=10,
                                       input_threshold=0.1,
                                       confidence_threshold=0.25,
                                       max_num=200)
    return detected_faces, input_image

def show_faces(detected_faces, img, is_log=False, is_picture=True):
    i = 0
    face_info="Face Info:\n"
    for face in detected_faces:
        face_info = face_info + f"Detect face-{i}: {face[0]},{face[1]},{face[2]},{face[3]}; score: {face[4]}" + "\n"
        # 绘制 Head 框
        l, t, r, b = map(int, face[:4])  # 将坐标转换为整数
        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)
        # 绘制 ID 文本
        cv2.putText(img, f"ID: {i}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        i += 1

    if is_log:
        logging.info(face_info)
    else:
        print(f"Detect face num: {len(detected_faces)}")

    if is_picture:
        cv2.namedWindow("Detected Faces", cv2.WINDOW_NORMAL)
        # if too big
        #cv2.resizeWindow("Detected Faces", 1920, 1080)
        cv2.imshow("Detected Faces", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    weight_path = "./S3FD_pytorch.pth"
    extractor = S3FDExtractor(weight_path)

    img_path = "./images/demo2.jpeg"  #"demo0.jpeg","demo1.jpg","demo2.jpeg"

    detected_faces, img = detect_face(extractor, img_path)

    show_faces(detected_faces, img, True, True)

