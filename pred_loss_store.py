import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argument
import time
import csv

args, unknown = argument.parse_args()

print("Loading dataset...")
start = time.time()
storage_base_path = './results/6_3_4/'
# Load dataset
# train_set = torch.load("./data/processed/{}_train.pt".format(args.dataset))
# valid_set = torch.load("./data/processed/{}_valid.pt".format(args.dataset))
# test_set = torch.load("./data/processed/{}_test.pt".format(args.dataset))


train_set = torch.load("./data/processed/{}_train_toy2.pt".format(args.dataset))
valid_set = torch.load("./data/processed/{}_train_toy2.pt".format(args.dataset))
test_set = torch.load("./data/processed/{}_train_toy2.pt".format(args.dataset))

print("Dataset Loaded! ({:.4f} sec)".format(time.time() - start))
from models.CGIB import CGIB

device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"

model = CGIB(device=device, num_step_message_passing=args.message_passing, dataset=args.dataset).to(device)
# 步骤1：加载保存的模型
# checkpoint = torch.load('model_to_be_analysis.pth')
checkpoint = torch.load('model_6_3_4.pth', map_location=torch.device('cpu'))
# torch.load('model_to_be_analysis2.pth', map_location=lambda storage, loc: storage.cuda(4))

# 提取模型状态字典和额外信息
model.load_state_dict(checkpoint['model_state_dict'], False)
perplexity_std_per_epoch = checkpoint['extra_info']['perplexity_std_per_epoch']
perplexity_mean_per_epoch = checkpoint['extra_info']['perplexity_mean_per_epoch']
vq_loss_per_epoch = checkpoint['extra_info']['vq_loss_per_epoch']
loss_per_epoch = checkpoint['extra_info']['loss_per_epoch']
pred_loss_per_epoch = checkpoint['extra_info']['pred_loss_per_epoch']

# 步骤2：获取 embedding 层
embedding_layer = model.vq.embedding

# 步骤3：提取 embedding 的数据
embedding_weights = embedding_layer.weight.data.cpu().numpy()
# 打印 embedding 数据
print("Embedding weights:")
print(embedding_weights)

# 步骤4：使用 t-SNE 进行可视化
tsne = TSNE(n_components=2, random_state=42)
embedding_tsne = tsne.fit_transform(embedding_weights)

# 绘制 t-SNE 可视化结果
plt.figure(figsize=(10, 8))
plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], marker='.')
plt.title('t-SNE Visualization of Embedding Layer')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# 保存图片
plt.savefig(storage_base_path + 'tsne_visualization.png')
plt.show()

# 提取embedding在t-SNE降维后的所有坐标
embedding_tsne_all_coords = embedding_tsne.tolist()

# 输出embedding在t-SNE降维后的所有坐标
print("Embedding t-SNE Coordinates:")
for i, coord in enumerate(embedding_tsne_all_coords):
    print(f"Point {i + 1}: ({coord[0]}, {coord[1]})")


# 可视化 perplexity_std_per_epoch
plt.figure(figsize=(10, 6))
plt.plot(perplexity_std_per_epoch, label='Perplexity Std')
plt.title('Perplexity Std per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Perplexity Std')
plt.legend()

# 保存图片
plt.savefig(storage_base_path + 'perplexity_std_per_epoch.png')
plt.show()

# 可视化 perplexity_mean_per_epoch
plt.figure(figsize=(10, 6))
plt.plot(perplexity_mean_per_epoch, label='Perplexity Mean')
plt.title('Perplexity Mean per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Perplexity Mean')
plt.legend()

# 保存图片
plt.savefig(storage_base_path + 'perplexity_mean_per_epoch.png')
plt.show()

# 可视化 vq_loss_per_epoch
plt.figure(figsize=(10, 6))
plt.plot(vq_loss_per_epoch, label='VQ Loss')
plt.title('VQ Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('VQ Loss')
plt.legend()

# 保存图片
plt.savefig(storage_base_path + 'vq_loss_per_epoch.png')
plt.show()

# 可视化 loss_per_epoch
plt.figure(figsize=(10, 6))
plt.plot(loss_per_epoch, label='Total Loss')
plt.title('Total Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.legend()

# 保存图片
plt.savefig(storage_base_path + 'loss_per_epoch.png')
plt.show()

# 可视化 pred_loss_per_epoch
plt.figure(figsize=(10, 6))
plt.plot(pred_loss_per_epoch, label='Pred Loss')
plt.title('Pred Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Pred Loss')
plt.legend()

# 保存图片
plt.savefig(storage_base_path + 'pred_loss_per_epoch.png')
plt.show()



# 保存到 CSV 文件
csv_file_path = storage_base_path + "embedding_tsne_coords.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Point', 't-SNE Component 1', 't-SNE Component 2'])
    for i, coord in enumerate(embedding_tsne_all_coords):
        writer.writerow([i + 1, coord[0], coord[1]])

print(f"Embedding t-SNE coordinates saved to: {csv_file_path}")

# 保存 Perplexity Std per Epoch 到 CSV 文件
perplexity_std_csv_file_path = storage_base_path + "perplexity_std_per_epoch.csv"
with open(perplexity_std_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Perplexity Std'])
    for epoch, value in enumerate(perplexity_std_per_epoch):
        writer.writerow([epoch + 1, value])

print(f"Perplexity Std per Epoch saved to: {perplexity_std_csv_file_path}")

# 保存 Perplexity Mean per Epoch 到 CSV 文件
perplexity_mean_csv_file_path = storage_base_path + "perplexity_mean_per_epoch.csv"
with open(perplexity_mean_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Perplexity Mean'])
    for epoch, value in enumerate(perplexity_mean_per_epoch):
        writer.writerow([epoch + 1, value])

print(f"Perplexity Mean per Epoch saved to: {perplexity_mean_csv_file_path}")

# 保存 VQ Loss per Epoch 到 CSV 文件
vq_loss_csv_file_path = storage_base_path + "vq_loss_per_epoch.csv"
with open(vq_loss_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'VQ Loss'])
    for epoch, value in enumerate(vq_loss_per_epoch):
        writer.writerow([epoch + 1, value])

print(f"VQ Loss per Epoch saved to: {vq_loss_csv_file_path}")

# 保存 Total Loss per Epoch 到 CSV 文件
total_loss_csv_file_path = storage_base_path + "total_loss_per_epoch.csv"
with open(total_loss_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Total Loss'])
    for epoch, value in enumerate(loss_per_epoch):
        writer.writerow([epoch + 1, value])

print(f"Total Loss per Epoch saved to: {total_loss_csv_file_path}")


# 保存 Pred Loss per Epoch 到 CSV 文件
pred_loss_csv_file_path = storage_base_path + "pred_loss_per_epoch.csv"
with open(pred_loss_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Pred Loss'])
    for epoch, value in enumerate(pred_loss_per_epoch):
        writer.writerow([epoch + 1, value])

print(f"Pred Loss per Epoch saved to: {pred_loss_csv_file_path}")
