# 卷积网络滤波器

## 简介:

    双卷积网络，分焦散光子和全局光子分别处理

## 依赖项:

    [Common]:
        1.Python 3
        2.TensorFlow
        3.OpenCV
        4.Scipy
        5.Pillow
        6.Colorama

    [Windows]:
        *matplotlib

## 运行程序:

    [run] main.py

## 参数配置:

    config.json

    ·卷积网络结构
        [conv_size] 各层卷积核大小
        [sfeatures] 场景空间特征数量
        [ifeatures] 图像空间特征数量
        [loss_func] 损失函数
        [optimizer] 优化器
        [active_func] 各层激活函数
        [weights_shape] 各层卷积核数量

    ·程序运行参数
        [mode] 运行模式，分infer和train两种
        [batch_n] 训练时切片数量
        [batch_h] 训练时切片高度
        [batch_w] 训练时切片宽度
        [cnn_name] 子网络名称
        [loss_func] 损失函数
        [max_round] 训练次数
        [learning_rate] 学习率
        [save_round] 迭代一定步数后保存

    ·程序输入数据
        [test_data] infer模式下的输入数据
        [train_data] train模式下的输入数据
        [ground_truth] 理想无噪图

    ·存储相关
        [model_path] checkpoint存储路径
        [ckpt_name] checkpoint文件名
        [meta_name] checkpoint相关元数据

    ·训练效果显示
        [plot_width] 显示窗口宽度
        [plot_height] 显示窗口高度

    ·其它
        [cache_size] 缓存大小
        [log_file] 日志输出文件

## 模块说明:

	·main.py
		程序入口，创建各类对象
	·mainproc.py
		程序主逻辑，指定任务流程
	·iosched.py
		场景文件的加载与切分
	·cache.py
		缓存场景数据
	·cnn.py
		构建卷积神经网络
	·filter.py
		进行图像滤波
	·irender.py
		Windows上绘制折线图
	·proc.py
		抽象任务处理类
	·prop.py
		存储程序所有参数
	·shared.py
		存储全局共享变量
	·singleton.py
		单例模式类
	·utils.py
		提供文件读写，图像显示等通用功能
	·test.py
		读取run.log中的误差数据以折线图显示
