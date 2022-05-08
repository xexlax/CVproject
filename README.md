# Project 3

case1、case2、case3 各给出 3 组大豆植物同一区域沿同一方向、按一定层厚的的连续细胞组织切片。离区标注以 JSON 文件形式给出。

`process.py`给出了 JSON 标注可视化的一个示例程序。

组长的补充

将数据集中的3个case文件夹放在此目录下以运行。
`count.py`给出计数程序

`separate.py`运行生成离区分割图像，将黑白mask图像存储在./predict目录（当前已做去噪声处理）

`evaluation.py`对分割结果进行量化评价并生成可视化对比  
使用时需在主函数下定义好文件目录以便运行:  
`label_path`：通过process程序生成的标准分割结果
`predict_path`：小组得出的分割结果  
`visual_path`：标准分割结果的可视化标注
