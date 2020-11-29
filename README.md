# Gauss-Conv

-- 1.自定义卷积核（Gauss）
-- 2.用于演示padding 0 在特征图边缘引起的现象 ，可以看出边缘特征受padding 0 影响，从而会影响在边缘处的小目标的检测
-- 3.out.jpg 与 out2.jpg的区别在于bias大小 bias如果有较大的负值则会出现out2.jpg的情况
-- 4.centernet训练过程中可视化特征图，观察到了特征图有out2.jpg类似情况
