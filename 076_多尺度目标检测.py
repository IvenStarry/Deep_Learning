import torch
from d2l import torch as d2l

# * 多尺度锚框
# 为了减少锚框数量，避免一些重复锚框，用较小的锚框可以检测小的物体，采样更多区域，用较大的锚框可以检测较大的物体，采样更少区域
img = d2l.plt.imread('related_data/cat_dog.png')
h, w = img.shape[:2]
print(h, w)

def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 生成特征图 前两个维度上的值不影响输出 
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    # 在特征图上每个像素点中心生成锚框
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    # anchors[0] 就是batchsize 
    # 乘以bbox_scale 就是在还原锚框在原图像的比例，以原图像的尺寸去还原特征图上的锚框，从而完成采样操作
    d2l.show_bboxes(d2l.plt.imshow(img).axes, anchors[0] * bbox_scale)
    d2l.plt.show()

# 探测小目标 假设特征图长宽是4
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
# 探测中目标 假设特征图长宽是2
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
# 探测大目标 假设特征图长宽是1
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])