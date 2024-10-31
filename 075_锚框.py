import torch
from d2l import torch as d2l

# * 生成多个锚框
'''
对于每一个像素，生成高宽都不同的锚框，锚框宽度w*s*sqrt(r) 这里w，h是图像宽和高，s是缩放比，r是宽高比
我们设置许多s取值 s1,s2...sm 和许多r取值 r1,r2...rm，为了简化样本个数，我们只考虑包含r1或s1的组合
即(s1, r1),(s1, r2),...(s1, rm),(s2, r1),(s3, r1)...(sm, r1)
'''
def multibox_prior(data, sizes, ratios): # data就是input
    in_height, in_width = data.shape[-2:] # 图像高和宽，像素点个数
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # 锚框个数 如上所述的组合个数
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素中心，设置偏移量，因为一个像素得分高宽均为1，因此偏移至中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height # 在y轴上缩放步长
    steps_w = 1.0 / in_width # 在x轴上缩放步长

    # 生成所有锚框的中心点
    '''
    我们希望锚框的中心位置对准像素的中心而非像素的左上角
    torch.arange(in_height, device=device) + offset_h 找到每一个像素点的中心坐标

    通过归一化操作，可以避免因输入图像尺寸变换而对结果的影响，也方便torch框架进行运算
    (torch.arange(in_height, device=device) + offset_h) * steps_h
    '''
    ceter_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    ceter_w = (torch.arange(in_width, device=device) + offset_w) * steps_w

    '''
    torch.meshgrid() 接受多个一维张量作为输入，并根据指定的索引模式 生成相应的多维网格张量。
    当 indexing='ij' （默认）时，第一个输入张量沿着行方向扩展，第二个输入张量沿着列方向扩展，行数是第一个输入张量的元素个数，列数是第二个输入张量的元素个数
    当 indexing='xy' 时，第一个输入张量沿着列方向扩展，第二个输入张量沿着行方向扩展，行数是第二个输入张量的元素个数，列数是第一个输入张量的元素个数
    
    通过meshgrid操作可以得到每一个锚框的中心点 应该在y轴和x轴上的偏移，拓展成了一个矩阵，shape(in_height, in_width)
    
    test.py
    import torch
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5])
    # 默认 indexing='ij'
    xx, yy = torch.meshgrid(x, y)
    print(xx)
    print(yy)
    # indexing='xy'
    xx_xy, yy_xy = torch.meshgrid(x, y, indexing='xy')
    print(xx_xy)
    print(yy_xy)
    '''
    shift_y, shift_x = torch.meshgrid(ceter_h, ceter_w, indexing='ij')
    # 拉成一维
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成每个像素点上 `boxes_per_pixel` 个锚框的高和宽
    # 将s和r组合 取出r1和s1,s2...sm s1和r2,r3...rm做组合操作
    # 为了适应不同图像的宽高比，消除原图像wh的影响，因此进行了归一化操作，
    '''
    先不看 in_height / in_width。上面解得的归一化后的公式与代码所写的一模一样。
    代码中的和就是锚框归一化后的宽高（此时消除了原图像 w 和 h 的影响，
    可以认为，r 所代表的宽高比就是此时锚框的宽高比，r=1 时，是一个正方形锚框，也即此时和的值是一样的）。
    但是呢，由于我们显示的时候需要乘以图像的实际宽高,所以，乘后的锚框实际宽高比就不是 1 了，所以才要乘以 in_height / in_width，
    作用就是抵消乘以实际图像长宽后 r 会改变的问题，当然这样做存粹是为了显示方便（也让你误以为 r 是指锚框的宽高比），
    带来的副作用就是，锚框的实际面积就不再是原始图像的。

    由于实际在我们进行目标检测时，特征图长和宽都是相同的，比如 (19, 19)、(7, 7)，
    所以 in_height / in_width 恒等于 1，因此对于实际的使用并不会带来副作用。
    但此时，如果要将锚框显示出来，归一化后的锚框再乘以图像实际长宽后，所显示的锚框的长宽比会改变。
    in_height / in_width 这部分失效了。好消息是，面积是原图的，又符合定义了。
    '''

    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # 获得半高和半宽
    '''
    torch.stack((-w, -h, w, h))  计算了相较锚点中心的偏移量，方便计算(x1, y1, x2, y2)

    .T 是将偏移量的形状进行转置，使得每行对应一个锚框的四个边界偏移量 (xmin, ymin, xmax, ymax)

    torch.repeat:输入一维张量，参数为两个(m,n)，即表示先在列上面进行重复n次，再在行上面重复m次，输出张量为二维
    repeat(in_height * in_width, 1)  因为每一个像素点都需要计算偏移，所以进行复制

    最后除以2 算得距离每一个锚框中心偏移量
    '''
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 
    '''
    torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1) 
    得到形状为 [in_height * in_width, 4] 的矩阵 每个像素点中心点的偏移
    .repeat_interleave(boxes_per_pixel, dim=0)：将每个像素点的中心坐标重复 boxes_per_pixel 次
    在dim=0上进行复制，这个一个像素点后紧跟着着这个像素点 boxes_per_pixel 个的锚框的中心偏移

    test.py
    a = torch.randn(2,2)
    print(a,a.repeat_interleave(3,dim=0)) # 重复三次
    output:
    tensor(
        [[-0.4402,  0.8147],
        [ 0.0875, -1.5945]])
    tensor(
        [[-0.4402,  0.8147],
        [-0.4402,  0.8147],
        [-0.4402,  0.8147],
        [ 0.0875, -1.5945],
        [ 0.0875, -1.5945],
        [ 0.0875, -1.5945]])
    '''
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    # 结合像素中心坐标和锚框偏移 每个锚框的四角坐标 (xmin, ymin, xmax, ymax)，也就是锚框的实际坐标
    output = out_grid + anchor_manipulations
    # 添加了一个维度 batch_size , shape [1, in_height * in_width * boxes_per_pixel, 4]
    return output.unsqueeze(0)

# 查看返回形状
img = d2l.plt.imread('related_data/cat_dog.png')
h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)

# 访问(200, 200)中心的第一个锚框
boxes = Y.reshape(h, w, 5, 4)
print(boxes[200, 200, 0, :])

# 显示以图像中某个像素为中心的所有锚框
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

# 以100，100为中心的所有锚框
d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[100, 100, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
d2l.plt.show()

# * 交并比IoU
#@save
def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    # None的作用增加了一个维度 便于广播
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # 使用 .clamp(min=0) 是为了处理可能的负值情况（即两个框没有重叠时），确保交集的宽和高至少为 0。
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    # torch.nonzero() 返回一个包含输入 input 中非零元素索引的张量.输出张量中的每行包含 input 中非零元素的索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    # 若为True，则保留当前位置的值，若为False，则舍弃
    box_j = indices[max_ious >= iou_threshold]
    # ! 这一步就是保底操作，能用的锚框都可以有一个最接近的真实框对应，小于阈值的直接舍弃，不给这些锚框对应标号
    anchors_bbox_map[anc_i] = box_j

    # 用于标记需要忽略的列（真实框）置-1
    col_discard = torch.full((num_anchors,), -1)
    # 用于标记需要忽略的行（锚框）置-1
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        # 全局最大值索引，是标量
        max_idx = torch.argmax(jaccard)
        # 确定真实框索引和锚框索引
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        # 更新分配映射
        # ! 这一步是图中的表格算法操作 也是确保每一个真实框都至少有一个对应锚框
        anchors_bbox_map[anc_idx] = box_idx
        # 将本行本列置0
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        # 调用 offset_boxes 计算锚框到真实边界框的偏移量，并乘以 bbox_mask 以掩盖无效锚框。
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
d2l.plt.show()

labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))

# * 使用非极大值抑制预测边界框
def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

# NMS按降序对置信度排序返回索引
def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
d2l.plt.show()

output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(output)

fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
d2l.plt.show()