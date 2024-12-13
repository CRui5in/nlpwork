import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(texts,
                                   truncation=True,
                                   padding=True,
                                   max_length=max_length,
                                   return_tensors='pt')

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def load_data(file_path):
    """加载文本数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return [text.strip() for text in texts if text.strip()]


def train_model(model, train_loader, val_loader, device, num_epochs=3):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # 将数据移到GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)

            # 使用最后一层的隐藏状态作为词向量
            last_hidden_states = outputs.last_hidden_state

            # 这里使用MLM损失作为示例
            # 实际应用中可以根据具体任务修改损失函数
            loss = torch.mean((last_hidden_states * attention_mask.unsqueeze(-1)).pow(2))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)

                last_hidden_states = outputs.last_hidden_state
                loss = torch.mean((last_hidden_states * attention_mask.unsqueeze(-1)).pow(2))
                val_loss += loss.item()

        logger.info(f'Epoch {epoch + 1}/{num_epochs}:')
        logger.info(f'Average training loss: {total_loss / len(train_loader)}')
        logger.info(f'Average validation loss: {val_loss / len(val_loader)}')

        model.train()


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # 加载tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    model.to(device)

    # 加载数据
    texts = load_data('zh.txt')  # 替换为你的文本文件路径

    # 划分训练集和验证集 (4:1)
    train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)

    # 创建数据集
    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # 训练模型
    train_model(model, train_loader, val_loader, device)

    # 保存模型
    output_dir = './model'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f'Model saved to {output_dir}')


if __name__ == '__main__':
    main()