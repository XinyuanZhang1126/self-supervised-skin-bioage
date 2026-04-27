# GitHub 上传执行计划

## 仓库信息
- **仓库名**: `self-supervised-skin-bioage`
- **可见性**: 公开 (public)
- **排除项**: `.gitignore` 已创建，排除了 `*.pth`, `*.pt`, `data/`, `outputs/` 等大文件

## 执行命令

在项目根目录 (`/225045037/rf-detr-seg-local`) 中依次执行以下命令：

```bash
# 1. 初始化 Git 仓库
git init

# 2. 配置 Git 用户信息（如未配置过）
git config user.email "your-email@example.com"
git config user.name "Your Name"

# 3. 添加所有文件并首次提交
git add .
git commit -m "Initial commit: RF-DETR segmentation fine-tuning project"

# 4. 创建 GitHub 公开仓库并关联远程（需先安装 GitHub CLI）
#    或手动在 https://github.com/new 创建仓库后执行：
git remote add origin https://github.com/YOUR_USERNAME/self-supervised-skin-bioage.git

# 5. 推送到 GitHub
git branch -M main
git push -u origin main
```

## 备选：使用 GitHub CLI 一键创建仓库

如果已安装 `gh` CLI：

```bash
gh auth login
gh repo create self-supervised-skin-bioage --public --source=. --push
```

## .gitignore 已排除的内容
- Python 缓存 (`__pycache__`, `*.pyc`)
- 虚拟环境 (`.venv`, `venv/`)
- Jupyter 检查点 (`.ipynb_checkpoints`)
- 模型权重 (`*.pth`, `*.pt`, `*.ckpt`)
- 训练输出 (`outputs/`)
- 数据集 (`data/`)
