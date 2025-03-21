import cv2
import numpy as np
import os

# jpg 或 png 图像的读取
def read_images_from_dir(dir, extensions=("jpg", "png")):
    images = []
    for filename in os.listdir(dir):
        if filename.lower().endswith(extensions):
            file_path = os.path.join(dir, filename)
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
    return images


# **************************第一部分：关键点检测**************************

# Harris角点检测，返回一组角点坐标
def harris_corner_detection(image, k=0.04, threshold=0.04):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # 利用Sobel算子计算图像的梯度
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度的平方和乘积
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    # 使用高斯核进行平滑
    Ix2 = cv2.GaussianBlur(Ix2, (3, 3), 1)
    Iy2 = cv2.GaussianBlur(Iy2, (3, 3), 1)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 1)
    # 计算Harris角点响应函数
    det = Ix2 * Iy2 - Ixy ** 2
    trace = Ix2 + Iy2
    response = det - k * (trace ** 2)
    # 通过阈值筛选来实现非极大值抑制
    keypoints = np.argwhere(response > threshold * response.max())
    keypoints = keypoints[:, [1, 0]]  # 转换为(x, y)格式
    return keypoints

# 绘制角点
def draw_keypoints(image, keypoints, output_path):
    image_with_keypoints = image.copy()
    for x, y in keypoints:
        cv2.circle(image_with_keypoints, (x, y), 2, (0, 0, 255), -1)
    cv2.imwrite(output_path, image_with_keypoints)


# **********************第二部分：特征表示与特征匹配**********************

# 返回SIFT特征描述子
def sift_descriptor(image, keypoints):
    sift = cv2.SIFT_create()
    keypoints_cv = [cv2.KeyPoint(x, y, 1) for x, y in keypoints]
    # 进行了16轮的DoG处理，返回descriptor，即n（关键点个数）个128维的向量列表
    _, descriptors = sift.compute(image, keypoints_cv)
    return descriptors

# 返回HOG特征描述子
def hog_descriptor(image, keypoints):
    hog = cv2.HOGDescriptor()
    descriptors = []
    for x, y in keypoints:
        x1 = max(0, x - 32)
        y1 = max(0, y - 64)
        x2 = min(image.shape[1], x + 32)
        y2 = min(image.shape[0], y + 64)
        # (x1,y1)是局部区域左上角的坐标
        # (x2,y2)是局部区域右下角的坐标
        # 将图像的一个局部传入HoG描述子中进行计算
        area = image[y1:y2,x1:x2]
        area = cv2.resize(area, (64, 128))
        descriptor = hog.compute(area)
        descriptors.append(descriptor)
    return np.array(descriptors)

# 特征匹配
def match_features(descriptors1, descriptors2, threshold=0.7):
    # 传入两个特征描述子，用相对阈值进行筛选
    bf = cv2.BFMatcher(crossCheck=True)
    # 调用BFMatcher实现根据欧氏距离进行描述子匹配
    matches = bf.match(descriptors1, descriptors2)
    good_matches = []
    max_match = 0
    for match in matches:
        if match.distance > max_match:
            max_match = match.distance
    for match in matches:
        if (max_match - match.distance) < max_match * threshold:
            good_matches.append(match)
    return good_matches


# *********************第三部分：图像矩阵变换与图像拼接********************

# 使用RANSAC计算仿射变换矩阵
def ransac_affine_transform(keypoints1, keypoints2, matches, iterations=1000, threshold=3):
    src_pts = np.float32([keypoints1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=iterations)
    return M

# 两幅图像的拼接
def stitch_images(img1, img2, M):
    # 获取图像的尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # 计算变换后的图像的边界
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    transformed_corners1 = cv2.transform(corners1, M)
    transformed_corners2 = corners2
    # 计算拼接图像的边界
    all_corners = np.concatenate((transformed_corners1, transformed_corners2), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    # 调整变换矩阵以考虑偏移
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]])
    M = np.vstack((M, [0, 0, 1]))  # 将 M 转换为 3x3 矩阵
    M = H_translation.dot(M)
    M = M[:2]  # 转换回 2x3 矩阵
    # 变换图像
    output_img = cv2.warpAffine(img1, M[:2], (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = img2
    return output_img


# *********************第四部分：全景图拼接流程（主函数）******************
# 全景图拼接
def panorama_stitching(images, output_dir):
    # 传入一组图像列表以及输出的文件夹的路径
    # 拼接算法产生的中间过程图像都会存放在输出文件夹中
    if not images:
        raise ValueError("No images provided for stitching")
    panorama = images[0]
    for i in range(1, len(images)):
        # 提取角点
        keypoints1 = harris_corner_detection(panorama)
        keypoints2 = harris_corner_detection(images[i])
        # 绘制角点图像
        draw_keypoints(panorama, keypoints1, f"{output_dir}/keypoints{i}_1.jpg")
        draw_keypoints(images[i], keypoints2, f"{output_dir}/keypoints{i}_2.jpg")
        # 提取特征描述子
        descriptors1 = hog_descriptor(panorama, keypoints1)
        descriptors2 = hog_descriptor(images[i], keypoints2)
        # 特征匹配
        matches = match_features(descriptors1, descriptors2)
        # 绘制匹配结果
        image_sift_matches = cv2.drawMatches(panorama, [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints1],
                                             images[i], [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints2],
                                             matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"{output_dir}/matches_{i}.jpg", image_sift_matches)
        # 计算仿射变换矩阵
        M = ransac_affine_transform(keypoints1, keypoints2, matches)
        # 拼接图像
        panorama = stitch_images(panorama, images[i], M)
        cv2.imwrite(f"{output_dir}/panorama_{i}.jpg", panorama)
    return panorama


# ********************************应用实例*********************************

def main():
    if not os.path.exists('middle_process_images'):
        os.makedirs('middle_process_images')
    # 读取/image文件夹下所有的图像
    dir = "images"
    images = read_images_from_dir(dir)
    # 拼接全景图
    panorama = panorama_stitching(images,'middle_process_images')
    cv2.imwrite('result.jpg', panorama)

if __name__ == '__main__':
    main()
