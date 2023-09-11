# Ziya Fastchat
基于Fastchat框架修改，以适配ziya专家模型的训练。

## RRHF

实现了[RRHF](https://arxiv.org/abs/2304.05302)。

目前支持batch_size=1的训练。

数据格式（每一行是一个样本）：

```python
{"prompt": "", "answers": ["a1", "a2", "a3"], "rewards": [0.625, 0.353515625, 0.2578125]}
```

其中，`a1`的reward对应0.625，以此类推。
