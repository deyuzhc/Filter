# Filter
卷积网络滤波器

简介:
双卷积网络，分焦散光子和全局光子分别处理

依赖项:

    [Common]:
        1.Python 3
        2.TensorFlow
        3.OpenCV
        4.Scipy
        5.Pillow
        6.Colorama

    [Windows]:
        *matplotlib

运行程序:

    [run] main.py

参数配置:

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

模块说明:
