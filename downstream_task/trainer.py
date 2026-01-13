"""
训练和评估逻辑
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
    
    Returns:
        tuple: (平均损失, 训练准确率)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / total, correct / total


def evaluate(model, loader, device):
    """
    评估模型
    
    Args:
        model: 模型
        loader: 数据加载器
        device: 计算设备
    
    Returns:
        tuple: (准确率, F1分数, AUC)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 取 TUM 的概率
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
        
    return acc, f1, auc


def train_model(model, train_loader, test_loader, epochs, lr, device):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 计算设备
    
    Returns:
        tuple: (最佳测试准确率, 最佳F1, 最佳AUC)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    
    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 评估
        val_acc, val_f1, val_auc = evaluate(model, test_loader, device)
        
        # 更新最佳结果
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_auc = val_auc
        
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
    
    return best_acc, best_f1, best_auc

