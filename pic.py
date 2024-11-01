import matplotlib.pyplot as plt



# 打开文本文件
R18_train_loss=[]
R18_val_loss = []
R18_train_ACC = []
R18_val_ACC = []
with open('save_resnet18_new1.txt', 'r') as file:
    # 按行读取文本内容
    for line in file.readlines():
        # 处理每一行
        # print(line.strip())  # 这里使用strip()函数去除行尾的换行符
        a = line.strip()
        b = a.split( )

        c = b[1].split(":")
        R18_train_loss.append(float(c[1]))

        c_1 = b[2].split(":")
        R18_val_loss.append(float(c_1[1]))

        c_2 = b[3].split(":")
        R18_train_ACC.append(float(c_2[1]))

        c_3 = b[4].split(":")
        R18_val_ACC.append(float(c_3[1]))

R50_train_loss=[]
R50_val_loss = []
R50_train_ACC = []
R50_val_ACC = []
with open('save_resnet50_new1.txt', 'r') as file:
    # 按行读取文本内容
    for line in file.readlines():
        # 处理每一行
        # print(line.strip())  # 这里使用strip()函数去除行尾的换行符
        a = line.strip()
        b = a.split( )

        c = b[1].split(":")
        R50_train_loss.append(float(c[1]))

        c_1 = b[2].split(":")
        R50_val_loss.append(float(c_1[1]))

        c_2 = b[3].split(":")
        R50_train_ACC.append(float(c_2[1]))

        c_3 = b[4].split(":")
        R50_val_ACC.append(float(c_3[1]))

Swin_train_loss=[]
Swin_val_loss = []
Swin_train_ACC = []
Swin_val_ACC = []
with open('save_swin_t_new1.txt', 'r') as file:
    # 按行读取文本内容
    for line in file.readlines():
        # 处理每一行
        # print(line.strip())  # 这里使用strip()函数去除行尾的换行符
        a = line.strip()
        b = a.split( )

        c = b[1].split(":")
        Swin_train_loss.append(float(c[1]))

        c_1 = b[2].split(":")
        Swin_val_loss.append(float(c_1[1]))

        c_2 = b[3].split(":")
        Swin_train_ACC.append(float(c_2[1]))

        c_3 = b[4].split(":")
        Swin_val_ACC.append(float(c_3[1]))

Vit_train_loss=[]
Vit_val_loss = []
Vit_train_ACC = []
Vit_val_ACC = []
with open('save_vit_new.txt', 'r') as file:
    # 按行读取文本内容
    for line in file.readlines():
        # 处理每一行
        # print(line.strip())  # 这里使用strip()函数去除行尾的换行符
        a = line.strip()
        b = a.split( )

        c = b[1].split(":")
        Vit_train_loss.append(float(c[1]))

        c_1 = b[2].split(":")
        Vit_val_loss.append(float(c_1[1]))

        c_2 = b[3].split(":")
        Vit_train_ACC.append(float(c_2[1]))

        c_3 = b[4].split(":")
        Vit_val_ACC.append(float(c_3[1]))

x = []
for i in range(30):
    x.append(i+1)
print(x)




# # 定义数据
#
#
# # 创建折线图，画两条线
# plt.plot(x, Vit_train_loss, label='train_loss')  # 第一条线
# plt.plot(x, Vit_val_loss, label='val_loss')  # 第二条线
#
# # 添加标题和标签
# plt.title('Vit loss')
# plt.xlabel('epoch')
# # plt.ylabel('Y-axis')
#
# # 添加图例
# plt.legend()
#
# # 保存图片
# plt.savefig('Vit_loss.png')
#
# #显示图形
# plt.show()

plt.plot(x, R50_train_ACC, label='train_acc')  # 第一条线
plt.plot(x, R50_val_ACC, label='val_acc')  # 第二条线

# 添加标题和标签
plt.title('ResNet-50 acc')
plt.xlabel('epoch')
plt.ylabel('acc')

# 添加图例
plt.legend()

# 保存图片
plt.savefig('pic1/resnet50_acc.png')

# 显示图形
plt.show()

# # 创建折线图，画两条线
# plt.plot(x, Vit_train_loss, label='Vit')  # 第一条线
# plt.plot(x, Swin_train_loss, label='Swin_Transformer')
# plt.plot(x, R50_train_loss, label='Resnet-50')
# plt.plot(x, R18_train_loss, label='Resnet-18')
# # 添加标题和标签
# plt.title('train loss')
# plt.xlabel('epoch')
# # plt.ylabel('acc')
#
# # 添加图例
# plt.legend()
#
# # 保存图片
# plt.savefig('pic1/train_loss.png')
#
# #显示图形
# plt.show()
