# 全景图拼接

本脚本使用传统的模式识别方法实现全景图的拼接

## 依赖库

pip install numpy
pip install opencv-python

## 使用方法

1. 将需要拼接的图像放入images文件夹内
需要注意，图像需要按照相关性顺序进行命名，防止两个相邻的图像没有关键信息交集的情况
2. python panorama.py 进行运行
3. middle_process_images文件夹中会存放图像匹配的中间图像，以便观察与调试
   包括关键点提取情况，图像匹配情况，中间拼接图像等
4. 最后会在该文件夹目录下生成 result.jpg 文件作为拼接结果
