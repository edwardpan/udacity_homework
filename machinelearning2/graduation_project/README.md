# 运行环境
项目基于ubuntu操作系统运行

从kaggle中下载数据文件后解压到当前目录，并创建preview文件夹，和saved_weights文件夹，形成如下的目录结构：
- data
  - imgs
    - train
      - c0
      - c1
      - c2
      - ......
    - test
    - preview
  - driver_imgs_list.csv
  - sample_submission.csv
- saved_weights

在requirements文件夹中有环境文件driver-ubuntu.yml

# 代码说明及大致运行时间
以下估计运行时间基于AWS中的p3.2xlarge服务器
data_augmenters.ipynb  图像拼接，约20分钟
inceptionv3.ipynb  训练InceptionV3模型及预测，约4-6小时
inceptionResNetV2.ipynb  训练InceptionResNetV2模型及预测，约4-6小时
resnet50.ipynb  训练ResNet50模型及预测，约4-6小时
densenet201.ipynb  训练DenseNet201模型及预测，约3-5小时
write_bottleneck.ipynb  导出特征向量，约1小时
mix_1.ipynb  融合模型训练及预测，约20分钟

split_valid.py  为验证集分割模块，将在各模型训练之前被调用
pred_view.py  为模型预测情况展示模块，将在各模型预测时使用，展示几张图片的预测结果

start_notebook.sh  为notebook启动脚本，使用`source activate tensorflow_p36`切换conda环境后可使用该脚本启动notebook，并在IP0.0.0.0监听
view_gpu.sh  为GPU状态查看启动脚本
submit_pred.sh  为提交预测结果的csv文件到kaggle的脚本，第一个参数为提交的文件路径，第二个参数为提交的注释。（需配置kaggle环境，SECRET_KEY）
# 模型最终结果
![](report_img/kaggel_score_1.jpg)
