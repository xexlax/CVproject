import os
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt


#  获取颜色字典
#  classNum 类别总数(含背景)
def color_dict(labelFolder, classNum):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if (len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if (len(colorDict) == classNum):
            break
    #  存储颜色的BGR字典，用于预测时的渲染结果
    colorDict_BGR = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位B,中3位G,后3位R
        color_BGR = [int(color[0: 3]), int(color[3: 6]), int(color[6: 9])]
        colorDict_BGR.append(color_BGR)
    #  转为numpy格式
    colorDict_BGR = np.array(colorDict_BGR)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1, colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY


def displayComparison(visual_path, predict_path, i):
    PredictList = os.listdir(predict_path)
    VisualList = os.listdir(visual_path)
    label1 = cv2.imread(visual_path + "/" + VisualList[i-1])
    prediction1 = cv2.imread(predict_path + "/" + PredictList[i])
    result1 = cv2.addWeighted(label1, 0.7, prediction1, 0.3, 10)
    cv2.namedWindow("Result", 0);
    cv2.resizeWindow("Result", 900, 600);
    cv2.imshow("Result", result1)
    cv2.waitKey(0)


def DisplayComparisons(visual_path, predict_path):
    PredictList = os.listdir(predict_path)
    VisualList = os.listdir(visual_path)
    pic_num = len(PredictList)
    plt.figure(figsize = (12, 20))
    plt.title("Comparison")
    for i in range(pic_num):
        label = cv2.imread(visual_path + "/" + VisualList[i])
        prediction = cv2.imread(predict_path + "/" + PredictList[i])
        result = cv2.addWeighted(label, 0.7, prediction, 0.3, 10)

        plt.subplot(6, 3, i + 1), plt.imshow(result), plt.title(i)
        plt.xticks([]), plt.yticks([])
    plt.show()


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        fWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return fWIoU

    def F1Score(self):
        # Precision = TP / (TP + FP), Recall = TP / (TP + FN)
        # F1-Score = 2 * Precision * Recall / (Precision + Recall)
        precision = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=1)
        recall = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=0)
        f1score = 2 * precision * recall / (precision + recall)
        return f1score

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # 返回混淆矩阵
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    label_path = './visualization'
    predict_path = './masks'
    visual_path = './visualization'

    PA = 0
    MIoU = 0
    F1_Score = 0

    labelList = os.listdir(label_path)
    PredictList = os.listdir(predict_path)
    VisualList = os.listdir(visual_path)
    pic_num = len(labelList)

    for i in range(pic_num):
        imgLabel = cv2.imread(label_path + "/" + labelList[i])
        imgPredict = cv2.imread(predict_path + "/" + PredictList[i])
        imgPredict = np.array(imgPredict)
        imgLabel = np.array(imgLabel, dtype='uint8')  # 可直接换成标注图片
        # 将数据转为单通道的图片
        imgLabel = cv2.cvtColor(imgLabel, cv2.COLOR_BGR2GRAY)
        imgPredict = cv2.cvtColor(imgPredict, cv2.COLOR_BGR2GRAY)
        imgPredict[imgPredict > 0] = 1
        imgLabel[imgLabel > 0] = 1

        metric = SegmentationMetric(2)  # 2表示有1个分类，有几个分类就填几
        print(imgPredict.shape, imgLabel.shape)
        metric.addBatch(imgPredict, imgLabel)

        PA = metric.pixelAccuracy()
        MIoU = metric.meanIntersectionOverUnion()
        F1_Score = metric.F1Score()
        FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
        print(metric.meanIntersectionOverUnion())


    print('像素准确率PA:            %.2f%%' % (PA * 100))
    print('平均交并比MIoU:           %.2f%%' % (MIoU * 100))
    print('权频交并比FWIoU:          %.2f%%' % (FWIoU * 100))
    # print('F1-score:          %.2f%%' % (F1_Score * 100))
    DisplayComparisons(visual_path, predict_path)
    displayComparison(visual_path, predict_path, 1)













