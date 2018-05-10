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
        7.eprogress

    [Windows]:
        *matplotlib

## 运行程序:

    [run] main.py

## 参数配置:

    config.json

    ·卷积网络结构
        [conv_size]
        [sfeatures]
        [ifeatures]
        [weights_shape]
        [active_func]

    ·程序运行参数
        [mode]
        [batch_n]
        [batch_h]
        [batch_w]
        [cnn_name]
        [loss_func]
        [max_round]
        [learning_rate]

    ·程序输入数据
        [test_data]
        [train_data]
        [ground_truth]

    ·训练结果存储
        [model_path]
        [ckpt_name]

    ·训练效果显示
        [plot_width]
        [plot_height]

    ·其它参数
        [cache_size]

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
	·test.py
		测试程序流程
	·utils.py
		提供文件读写，图像显示等通用功能
