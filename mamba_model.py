"""
基于Mamba模型的运动分类系统
数据集：包含EMG、加速度计、陀螺仪等传感器数据
任务：分类4种运动类型 - levelground, stair, treadmill, ramp
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from mamba_models import MambaClassifier
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ==================== 数据加载和预处理 ====================
class ActivityDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_and_preprocess_data(train_path='train_features.csv', test_path='test_features.csv', val_size=0.1):
    """加载并预处理数据"""
    print("正在加载数据...")
    # 读取训练集和测试集
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"训练集大小: {df_train.shape}")
    print(f"测试集大小: {df_test.shape}")
    print(f"训练集活动类型分布:\n{df_train['activity_type'].value_counts()}")
    print(f"测试集活动类型分布:\n{df_test['activity_type'].value_counts()}")
    
    # 分离特征和标签
    # 后3列是: file_name, activity_type, segment_index
    feature_columns = df_train.columns[:-3]
    X_train_full = df_train[feature_columns].values
    y_train_full = df_train['activity_type'].values
    X_test = df_test[feature_columns].values
    y_test = df_test['activity_type'].values
    
    print(f"\n特征列数: {len(feature_columns)}")
    
    # 标签编码
    label_encoder = LabelEncoder()
    y_train_full_encoded = label_encoder.fit_transform(y_train_full)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"\n类别映射:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"  {idx}: {label}")
    
    # 数据标准化 (使用训练集的统计量)
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    # 从训练集中划分出验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_scaled, y_train_full_encoded, 
        test_size=val_size, random_state=42, stratify=y_train_full_encoded
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test_scaled.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test_scaled, y_test_encoded), label_encoder, scaler

# ==================== 训练和评估 ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='训练')
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels

def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图已保存至: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存至: {save_path}")
    plt.close()

# ==================== 主程序 ====================
def main():
    # 配置参数
    config = {
        'train_path': 'train_features.csv',
        'test_path': 'test_features.csv',
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'd_model': 128,
        'n_layers': 4,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'dropout': 0.1,
        'val_size': 0.1,
    }
    
    print("=" * 60)
    print("基于Mamba模型的运动分类系统")
    print("=" * 60)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder, scaler = \
        load_and_preprocess_data(config['train_path'], config['test_path'], config['val_size'])
    
    # 创建数据加载器
    train_dataset = ActivityDataset(X_train, y_train)
    val_dataset = ActivityDataset(X_val, y_val)
    test_dataset = ActivityDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # 创建模型
    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    
    model = MambaClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        d_state=config['d_state'],
        d_conv=config['d_conv'],
        expand=config['expand'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                                 weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs']
    )
    
    # 训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'label_encoder': label_encoder,
                'scaler': scaler
            }, 'best_model.pth')
            print(f"✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发! {patience} 个epoch验证准确率未提升")
            break
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 加载最佳模型进行测试
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型...")
    print("=" * 60)
    
    checkpoint = torch.load('best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\n测试集结果:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.2f}%")
    
    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, 
                              target_names=label_encoder.classes_,
                              digits=4))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(test_labels, test_preds, label_encoder.classes_)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"模型已保存至: best_model.pth")
    print("=" * 60)

if __name__ == '__main__':
    main()
