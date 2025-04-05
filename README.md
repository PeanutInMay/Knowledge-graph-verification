# 多智能体知识图谱校验系统

这是一个基于大模型的知识三元组校验系统，用于检测并修正知识三元组中可能存在的幻觉问题。系统采用多智能体串联架构，通过三个智能体的协同工作，对知识三元组进行全面评估和修正。

## 系统架构

本项目包含两个版本的实现：
- **API版本** ([`demo.py`](app.py )): 调用远程大模型API
- **本地版本** (`app.py`): 使用本地加载的大模型

系统核心是一个三级智能体架构：
1. **智能体1(初步验证)**: 检查三元组中的头实体和尾实体是否真实存在
2. **智能体2(事实核查)**: 进行更深入的事实核查，评估实体存在的可信度
3. **智能体3(最终判断)**: 整合前两个智能体的结果，给出最终判断和修正建议

## 安装指南

1. 克隆本仓库
2. 安装所需依赖:

```bash
pip install -r requirements.txt
```


## 使用说明

### API版本（demo.py）

1. 修改 [`demo.py`](demo.py ) 中的API密钥:

```python
client = OpenAI(
    api_key="YOUR_API_KEY_HERE",  # 替换为你的API密钥
    base_url="https://api.deepseek.com"  # 或其他支持OpenAI兼容API的服务地址
)
```

2. 运行程序:

```bash
python demo.py
```

### 本地版本（app.py）

1. 修改 [`app.py`](app.py ) 中的模型路径:

```python
model_path = "YOUR_MODEL_PATH_HERE"  # 替换为你的本地模型路径
device = "cuda:0"  # 根据你的硬件情况调整
```

2. 运行程序:

```bash
python app.py
```

## 使用界面

启动后，系统提供了一个用户友好的Web界面：
1. 在输入框中填写三元组的头实体、关系和尾实体
2. 点击"开始校验"按钮
3. 系统将依次调用三个智能体进行校验
4. 修正结果区域将显示修正后的三元组
5. 可在"校验过程详情"部分查看详细的校验过程

## 示例三元组

- 头实体: "爱因斯坦", 关系: "发明了", 尾实体: "相对论"
- 头实体: "拿破仑", 关系: "发明了", 尾实体: "蒸汽机"
- 头实体: "马云", 关系: "创立了", 尾实体: "阿里巴巴"
- 头实体: "关羽", 关系: "是", 尾实体: "三国演义的角色"

## 注意事项

- **安全提示**: 本代码仓库中的API密钥和模型路径已被移除。使用前请替换为您自己的密钥和路径。
- **硬件要求**: 本地版本需要足够的GPU内存来加载大模型。
- **API使用**: API版本需要确保您有足够的API调用额度。

## 文件结构

- [`app.py`](app.py ): 本地大模型版本实现
- [`demo.py`](demo.py ): API调用版本实现
- [`requirements.txt`](requirements.txt ): 项目依赖库
