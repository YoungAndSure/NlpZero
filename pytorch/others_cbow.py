#! python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot

# for learn copy from : https://zhuanlan.zhihu.com/p/681543485

# 定义CBOW模型类，继承自nn.Module，是所有神经网络模块的基类
class CBOW(nn.Module):
    # 构造函数，定义模型初始化时需要的参数：词汇表大小和嵌入向量的维度
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()  # 调用父类的构造函数来进行初始化
        # 创建一个嵌入层，它将词汇表中的每个词映射到一个固定大小的嵌入向量
        # vocab_size指定了嵌入层的大小，即有多少个嵌入向量
        # embedding_dim指定了每个嵌入向量的维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 创建一个线性层，用于将嵌入向量转换回词汇表大小的输出
        # 输入维度是嵌入维度，输出维度是词汇表的大小，这样可以将嵌入向量映射回一个词的预测概率
        self.linear1 = nn.Linear(embedding_dim, vocab_size)
    
    # 定义模型的前向传播路径
    # context是输入的上下文词的索引
    def forward(self, context):
        # 使用嵌入层获取上下文词的嵌入向量，并计算这些向量的平均值
        # 这里的平均是在嵌入向量的维度上进行的，dim=0表示沿着批处理维度（如果有的话）或者列表的第一个维度求平均
        embeds = torch.mean(self.embeddings(context), dim=0)
        # 将平均后的嵌入向量通过线性层转换，以预测中心词
        out = self.linear1(embeds)
        # 使用log_softmax函数计算预测的概率分布的对数形式
        # 这里的dim=0参数确保概率是沿着正确的维度进行归一化的
        # 对数概率通常用于计算损失函数，因为它们在数值上更稳定
        log_probs = torch.log_softmax(out, dim=0)
        # 返回预测的对数概率
        return log_probs

# 创建模型实例
vocab_size = 4  # 假设的词汇表大小
embedding_dim = 10  # 嵌入维度
model = CBOW(vocab_size, embedding_dim)

# 创建一个示例输入
context = torch.tensor([0, 2, 3], dtype=torch.long)  # 示例上下文索引

# 执行一次前向传播
log_probs = model(context)

# 使用torchviz可视化模型
vis_graph = make_dot(log_probs, params=dict(model.named_parameters()))
vis_graph.render('cbow_model_visualization', format='png')  # 输出图像文件

# 定义词汇表
word_to_ix = {"hope": 0, "is": 1, "a": 2, "good": 3}
vocab_size = len(word_to_ix)

# 上下文窗口大小为2（一个目标词左右各一个词）
data = [
    (["hope", "is"], "a"),
    (["is", "a"], "good"),
    (["a", "good"], "hope"),
]

# 为训练数据创建上下文和目标张量
context_idxs = [[word_to_ix[w] for w in context] for context, target in data]
print(f'Context idxs: {context_idxs}')
target_idxs = [word_to_ix[target] for context, target in data]
print(f'Target idxs: {target_idxs}')

# 转换为PyTorch张量
context_tensors = [torch.tensor(x, dtype=torch.long) for x in context_idxs]
print(f'Context tensors: {context_tensors}')
target_tensors = [torch.tensor(x, dtype=torch.long) for x in target_idxs]
print(f'Target tensors: {target_tensors}')

# 设置模型参数
embedding_dim = 10

# 初始化模型、损失函数和优化器
model = CBOW(vocab_size, embedding_dim)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):
    total_loss = 0
    for context, target in zip(context_tensors, target_tensors):
        # 步骤 1. 准备数据
        context_var = context

        # 步骤 2. 清零梯度
        model.zero_grad()

        # 步骤 3. 运行前向传播
        log_probs = model(context_var)

        # 步骤 4. 计算损失函数
        loss = loss_function(log_probs.view(1,-1), target.view(-1))

        # 步骤 5. 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {total_loss}')

def predict_context(context, model, word_to_ix):
    # 将上下文词转换为对应的索引
    context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
    
    # 使用模型进行预测
    with torch.no_grad():  # 不需要计算梯度
        log_probs = model(context_idxs)
    
    # 找到概率最大的词的索引
    max_log_prob, max_idx = torch.max(log_probs, dim=0)
    
    # 将索引转换回词
    ix_to_word = {ix: word for word, ix in word_to_ix.items()}
    return ix_to_word[max_idx.item()]

# 假设模型已经训练完成，并且word_to_ix是我们的词汇表索引字典
# 举个例子，预测给定上下文["hope", "is"]的中心词
predicted_word = predict_context(["hope", "is"], model, word_to_ix)
print(f'Predicted center word: {predicted_word}')