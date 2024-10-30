import torch
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.plt.imread('related_data/cat_dog.png')
d2l.plt.imshow(img)
d2l.plt.show()

# * 边界框
# 定义两种表示间转换的函数
def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# 猫狗边界框
dog_bbox, cat_bbox = [20.0, 25.0, 180.0, 220.0], [185.0, 55.0, 290.0, 210.0]

boxes = torch.tensor((dog_bbox, cat_bbox))
print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

# 将边界框画出
def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(xy=(bbox[0], bbox[1]),
                            width=bbox[2] - bbox[0],
                            height=bbox[3] - bbox[1],
                            fill=False,
                            edgecolor=color,
                            linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show()