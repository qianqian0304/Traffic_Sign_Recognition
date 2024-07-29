# import cv2
# import os
# import numpy as np
# import pandas as pd
# from skimage.feature import hog
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
#
# # 读取 meta.csv 文件
# meta_df = pd.read_csv('D:/STUDY/NUS/Dataset_2/Meta.csv')
# # 创建标签到颜色的映射字典
# label_to_color = dict(zip(meta_df['ClassId'], meta_df['ColorId']))
#
# # 读取并处理数据集函数
# def read_and_process_images(csv_file, img_dir):
#     df = pd.read_csv(csv_file)
#     X_outer = []
#     hsv_outer = []
#     X_inner = []
#     hog_inner = []
#     y_labels = []
#
#     for index, row in df.iterrows():
#         img_path = os.path.join(img_dir, row['Path'])
#         label = row['ClassId']
#         y_labels.append(label)
#
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"Error reading image {img_path}")
#         else:
#             # 按照Roi坐标裁剪图像
#             roi = img[row['Roi.Y1']:row['Roi.Y2'], row['Roi.X1']:row['Roi.X2']]
#             temp_img = cv2.resize(roi, (48, 48))
#
#             h, w, _ = temp_img.shape
#             border_h = int(h * 0.20)
#             border_w = int(w * 0.20)
#
#             # 获取外围40%的部分
#             outer_img = np.zeros_like(temp_img)
#             outer_img[:border_h, :] = temp_img[:border_h, :]
#             outer_img[-border_h:, :] = temp_img[-border_h:, :]
#             outer_img[:, :border_w] = temp_img[:, :border_w]
#             outer_img[:, -border_w:] = temp_img[:, -border_w:]
#
#             # 获取内部60%的部分
#             inner_img = temp_img[border_h:-border_h, border_w:-border_w]
#
#             # 转换为HSV颜色空间并展平
#             hsv_outer.append(cv2.cvtColor(outer_img, cv2.COLOR_BGR2HSV).flatten())
#
#             # 转换为灰度图像
#             gray_inner_img = cv2.cvtColor(inner_img, cv2.COLOR_BGR2GRAY)
#             # 提取HOG特征
#             hog_features = hog(gray_inner_img, orientations=8, pixels_per_cell=(10, 10),
#                                cells_per_block=(1, 1), visualize=False)
#             hog_inner.append(hog_features)
#
#     return hsv_outer, hog_inner, y_labels
#
# # 读取训练数据
# hsv_train_outer, hog_train_inner, y_train = read_and_process_images('D:/STUDY/NUS/Dataset_2/Train.csv', 'D:/STUDY/NUS/Dataset_2/')
#
# # 读取测试数据
# hsv_test_outer, hog_test_inner, y_test = read_and_process_images('D:/STUDY/NUS/Dataset_2/Test.csv', 'D:/STUDY/NUS/Dataset_2/')
#
# # 创建颜色标签
# color_labels_train = [label_to_color[int(label)] for label in y_train]
# color_labels_test = [label_to_color[int(label)] for label in y_test]
#
# # 训练随机森林分类器（外围区域）
# rf_classifier_outer = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier_outer.fit(hsv_train_outer, color_labels_train)
#
# # 评估随机森林分类器（外围区域）
# y_pred_rf_outer = rf_classifier_outer.predict(hsv_test_outer)
# accuracy_rf_outer = accuracy_score(color_labels_test, y_pred_rf_outer)
# print(f"Color classification accuracy with RandomForest (outer region): {accuracy_rf_outer * 100:.2f}%")
#
# # SVM 分类器（内部区域）
# color_categories = ["red", "blue", "yellow", "white"]
# total_correct = 0
# total_samples = 0
#
# # 使用外围训练数据进行预测，以获得训练集的颜色标签
# y_pred_train_rf_outer = rf_classifier_outer.predict(hsv_train_outer)
#
# for color in range(4):
#     print(f"Training SVM classifier for {color_categories[color]} signs")
#
#     color_indices_train = np.where(y_pred_train_rf_outer == color)[0]
#     color_indices_test = np.where(y_pred_rf_outer == color)[0]
#
#     # 特殊处理 color == 2 的情况
#     if color == 2:
#         length = len(color_indices_test)
#         array = np.full(length, 12)
#         y_color_test = np.array(y_test)[color_indices_test]
#         accuracy = accuracy_score(y_color_test, array)
#         print(f"{color_categories[color].capitalize()} classification accuracy: {accuracy * 100:.2f}%")
#         total_correct += accuracy * len(y_color_test)
#         total_samples += len(y_color_test)
#         continue
#
#     X_color_hog_train = [hog_train_inner[idx] for idx in color_indices_train]
#     X_color_hog_test = [hog_test_inner[idx] for idx in color_indices_test]
#
#     y_train_int = [int(label) for label in y_train]
#     y_color_train = np.array(y_train_int)[color_indices_train]
#     y_color_test = np.array(y_test)[color_indices_test]
#
#     # 训练 SVM 分类器
#     svm_classifier_inner = SVC()
#     svm_classifier_inner.fit(X_color_hog_train, y_color_train)
#     y_pred_svm_inner = svm_classifier_inner.predict(X_color_hog_test)
#
#     accuracy = accuracy_score(y_color_test, y_pred_svm_inner)
#     print(f"{color_categories[color].capitalize()} classification accuracy: {accuracy * 100:.2f}%")
#     total_correct += accuracy * len(y_color_test)
#     total_samples += len(y_color_test)
#
# overall_accuracy = total_correct / total_samples
# print(f"\nOverall classification accuracy: {overall_accuracy * 100:.2f}%")
#
# # 绘制混淆矩阵
# unique_colors = sorted(set(color_labels_train))
# cm = confusion_matrix(color_labels_test, y_pred_rf_outer, labels=unique_colors)
#
# plt.figure(figsize=(10, 7))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title(f'Confusion Matrix (Accuracy: {accuracy_rf_outer * 100:.2f}%)')
# plt.colorbar()
# tick_marks = np.arange(len(unique_colors))
# plt.xticks(tick_marks, unique_colors, rotation=45)
# plt.yticks(tick_marks, unique_colors)
#
# thresh = cm.max() / 2.
# for i, j in np.ndindex(cm.shape):
#     plt.text(j, i, format(cm[i, j], 'd'),
#              horizontalalignment="center",
#              color="white" if cm[i, j] > thresh else "black")
#
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.tight_layout()
# plt.show()


#丢弃中间像素
import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 读取 meta.csv 文件
meta_df = pd.read_csv('D:/STUDY/NUS/Dataset_2/Meta.csv')
# 创建标签到颜色的映射字典
label_to_color = dict(zip(meta_df['ClassId'], meta_df['ColorId']))

# 读取并处理数据集函数
def read_and_process_images(csv_file, img_dir):
    df = pd.read_csv(csv_file)
    hsv_outer = []
    hog_inner = []
    y_labels = []

    for index, row in df.iterrows():
        img_path = os.path.join(img_dir, row['Path'])
        label = row['ClassId']
        y_labels.append(label)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image {img_path}")
        else:
            # 按照Roi坐标裁剪图像
            roi = img[row['Roi.Y1']:row['Roi.Y2'], row['Roi.X1']:row['Roi.X2']]
            # 计算需要缩放的大小以调整为正方形
            size = 48
            temp_img = cv2.resize(roi, (size, size))

            h, w, _ = temp_img.shape
            border_h = int(h * 0.20)
            border_w = int(w * 0.20)

            # 获取外围40%的部分
            top_part = temp_img[:border_h, :]  # 顶部
            bottom_part = temp_img[-border_h:, :]  # 底部
            left_part = temp_img[:, :border_w]  # 左侧
            right_part = temp_img[:, -border_w:]  # 右侧

            # 拼接外围部分，确保维度一致
            outer_img_vertical = np.vstack([top_part, bottom_part])  # 垂直拼接顶部和底部
            outer_img_vertical_flipped = np.rot90(outer_img_vertical, k=1)  # 将垂直拼接后的图像顺时针旋转90度

            outer_img_horizontal = np.hstack([left_part, right_part])  # 水平拼接左侧和右侧

            # 水平拼接旋转后的垂直拼接部分和水平拼接部分
            outer_img = np.hstack([outer_img_vertical_flipped, outer_img_horizontal])

            # 转换为HSV颜色空间并展平
            hsv_outer.append(cv2.cvtColor(outer_img, cv2.COLOR_BGR2HSV).flatten())

            # 获取内部60%的部分
            inner_img = temp_img[border_h:-border_h, border_w:-border_w]
            # 转换为灰度图像
            gray_inner_img = cv2.cvtColor(inner_img, cv2.COLOR_BGR2GRAY)
            # 提取HOG特征
            hog_features = hog(gray_inner_img, orientations=8, pixels_per_cell=(5, 5),
                               cells_per_block=(1, 1), visualize=False)
            hog_inner.append(hog_features)

    return hsv_outer, hog_inner, y_labels

# 读取训练数据
hsv_train_outer, hog_train_inner, y_train = read_and_process_images('D:/STUDY/NUS/Dataset_2/Train.csv', 'D:/STUDY/NUS/Dataset_2/')

# 读取测试数据
hsv_test_outer, hog_test_inner, y_test = read_and_process_images('D:/STUDY/NUS/Dataset_2/Test.csv', 'D:/STUDY/NUS/Dataset_2/')

# 创建颜色标签
color_labels_train = [label_to_color[int(label)] for label in y_train]
color_labels_test = [label_to_color[int(label)] for label in y_test]

# 训练随机森林分类器（外围区域）
rf_classifier_outer = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_outer.fit(hsv_train_outer, color_labels_train)

# 评估随机森林分类器（外围区域）
y_pred_rf_outer = rf_classifier_outer.predict(hsv_test_outer)
accuracy_rf_outer = accuracy_score(color_labels_test, y_pred_rf_outer)
print(f"Color classification accuracy with RandomForest (outer region): {accuracy_rf_outer * 100:.2f}%")

# SVM 分类器（内部区域）
color_categories = ["red", "blue", "yellow", "white"]
total_correct = 0
total_samples = 0

# 使用外围训练数据进行预测，以获得训练集的颜色标签
y_pred_train_rf_outer = rf_classifier_outer.predict(hsv_train_outer)

for color in range(4):

    print(f"Training SVM classifier for {color_categories[color]} signs")

    color_indices_train = np.where(y_pred_train_rf_outer == color)[0]
    color_indices_test = np.where(y_pred_rf_outer == color)[0]

    # 特殊处理 color == 2 的情况
    if color == 2:
        length = len(color_indices_test)
        array = np.full(length, 12)
        y_color_test = np.array(y_test)[color_indices_test]
        accuracy = accuracy_score(y_color_test, array)
        print(f"{color_categories[color].capitalize()} classification accuracy: {accuracy * 100:.2f}%")
        total_correct += accuracy * len(y_color_test)
        total_samples += len(y_color_test)
        continue

    X_color_hog_train = [hog_train_inner[idx] for idx in color_indices_train]
    X_color_hog_test = [hog_test_inner[idx] for idx in color_indices_test]

    y_train_int = [int(label) for label in y_train]
    y_color_train = np.array(y_train_int)[color_indices_train]
    y_color_test = np.array(y_test)[color_indices_test]

    # 训练 SVM 分类器
    svm_classifier_inner = SVC()
    svm_classifier_inner.fit(X_color_hog_train, y_color_train)
    y_pred_svm_inner = svm_classifier_inner.predict(X_color_hog_test)

    accuracy = accuracy_score(y_color_test, y_pred_svm_inner)
    print(f"{color_categories[color].capitalize()} classification accuracy: {accuracy * 100:.2f}%")
    total_correct += accuracy * len(y_color_test)
    total_samples += len(y_color_test)

overall_accuracy = total_correct / total_samples
print(f"\nOverall classification accuracy: {overall_accuracy * 100:.2f}%")

# 绘制混淆矩阵
unique_colors = sorted(set(color_labels_train))
cm = confusion_matrix(color_labels_test, y_pred_rf_outer, labels=unique_colors)

plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix (Accuracy: {accuracy_rf_outer * 100:.2f}%)')
plt.colorbar()
tick_marks = np.arange(len(unique_colors))
plt.xticks(tick_marks, unique_colors, rotation=45)
plt.yticks(tick_marks, unique_colors)

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

