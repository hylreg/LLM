# LLM

一个基础的Python项目模板，演示如何集成Silicon Flow(硅基流动)的大语言模型。

## 项目简介

这个项目展示了如何在Python项目中使用Silicon Flow(硅基流动)提供的大语言模型。Silicon Flow是一个提供多种大语言模型API服务的平台，兼容OpenAI API规范，使得集成变得非常容易。

## 项目结构

```
.
├── main.py                 # 主程序入口
├── pyproject.toml          # 项目配置文件
├── test_langchain/        # LangChain集成测试模块（已弃用）
└── README.md
```

## 快速开始

### 环境配置

1. 确保你已经安装了Python 3.10或更高版本
2. 设置环境变量:
   ```bash
   export SILICONFLOW_API_KEY=your_actual_api_key_here
   export SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1/  # 可选
   ```

### 安装依赖

使用`uv`安装项目依赖:
```bash
uv sync
```

### 运行项目

执行主程序:
```bash
python main.py
```

这将会调用Silicon Flow上的模型并显示结果。

## 更多信息

有关Silicon Flow的更多信息，请访问[SolarFlow官网](https://www.siliconflow.cn/)