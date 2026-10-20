# Helmet Detection — Custom YOLO Model
Trained for construction-site safety monitoring, delivered ready to run.

> Status: project scaffold. Metrics are placeholders — fill in only real results.

## Demo
[演示 GIF 待录]

## Use Case
工地场景检测作业人员是否佩戴安全帽。

## Dataset
- 来源：Roboflow Universe 公开数据集 [待填链接]
- 图片数：[待填]
- 类别：[按实际数据集填]

## Results
| mAP50 | Precision | Recall |
| --- | --- | --- |
| [待填真实结果] | [待填] | [待填] |

测试集与划分方式：[待填]

## Run It
pip install ultralytics
yolo predict model=best.pt source=your_image.jpg

## Deliverables
- best.pt 训练权重
- 推理脚本（单图 / 批量 / 视频）
- 本 README 使用说明
