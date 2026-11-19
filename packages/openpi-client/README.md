# 策略远程调用客户端

本模块负责 **LIBERO 仿真环境与策略推理服务之间的通信**，不承担模型训练任务。

- `websocket_client_policy.py`：通过 WebSocket 向策略服务端发送观测并接收动作预测。
- `image_tools.py`：图像编码与尺寸变换。
- `action_chunk_broker.py`：动作块管理。
- `base_policy.py`：客户端策略协议基础。
- `msgpack_numpy.py`：NumPy 数据序列化。

典型调用链：`examples/libero/main.py` → `openpi_client.websocket_client_policy` → 策略服务端 → VLA 推理。
