#!/bin/bash

# 创建项目
vue create frontend -p default

# 进入项目目录
cd frontend

# 安装依赖
npm install -D tailwindcss@latest postcss@latest autoprefixer@latest

# 初始化 Tailwind CSS
npx tailwindcss init -p

# 创建必要的目录和文件
mkdir -p src/components
mkdir -p src/assets

# 创建环境变量文件
echo "VITE_API_BASE_URL=http://localhost:8000
VITE_ENSEMBLE_API_URL=http://localhost:8001" > .env

# 启动开发服务器
npm run serve 