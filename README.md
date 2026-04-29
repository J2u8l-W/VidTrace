# VidTrace

VidTrace 是一个 AI 驱动的视频溯源与风险分析原型项目。  
项目聚焦于将高成本、低复现性的人工流程自动化，覆盖视频素材侵权检测、AIGC 视频识别、模型可追溯分析与风险用户挖掘等场景。

## 项目要解决的核心痛点

在真实业务中，视频内容治理通常面临以下问题：

- 素材来源复杂，人工核验路径长、耗时高。
- 分析流程依赖经验，难以标准化复现。
- 证据链分散，跨团队协作和复核成本高。

VidTrace 将多个算法模块整合为统一后端工作流，使“输入任务 -> 多模块推理 -> 结构化结果输出”形成闭环。

## Agent / AI 驱动构建成果

本项目采用“Agent 式任务编排 + 多模块协同推理”的核心逻辑：

1. **任务编排**：后端接收请求并路由到对应能力模块。  
2. **专用推理**：不同模块处理不同类型证据。  
   - `VideoMaterialsInfringement`：视频片段相似性匹配与拷贝区间定位  
   - `AGVRecognition`：基于 Transformer 的 AIGC 视频识别  
   - `ModelTraceability`：基于重构与反演的模型可追溯分析  
   - `RiskUserMining`：基于 GraphSAGE 的风险用户挖掘  
3. **结果聚合**：将中间结果整合为可复核、可机器处理的输出。

从方法论上看，该流程体现了**长链推理 + 多模块协作**：  
系统会先拆分子任务，再由模块并行或顺序执行，最后汇聚结果形成结论，显著降低人工排查成本并提升一致性。

## 仓库结构

```text
VidTrace/
├─ FlaskBackend/
│  ├─ main.py
│  ├─ requirements.txt
│  └─ CoreModule/
│     ├─ AGVRecognition/
│     ├─ ModelTraceability/
│     ├─ RiskUserMining/
│     └─ VideoMaterialsInfringement/
└─ WorkReport.pdf
```

## 快速开始

### 1) 环境要求

- Python 3.9 及以上
- 推荐使用支持 CUDA 的 GPU 环境进行训练/推理

### 2) 安装依赖

```bash
cd FlaskBackend
pip install -r requirements.txt
```

### 3) 启动服务

```bash
python main.py
```

默认地址：

- `http://0.0.0.0:8080/`

健康检查返回：

- `Response Successful!`

### 4) 接口说明

`FlaskBackend/main.py` 提供 `/predict` 接口（`multipart/form-data` 上传图片），返回模型推理后的结果图像。

## 核心模块说明

### VideoMaterialsInfringement

- 面向视频素材侵权检测，进行跨视频特征匹配与片段定位。
- 支持置信度阈值与 NMS 阈值控制。

### AGVRecognition

- 使用 Transformer 分类器识别生成式视频样本。
- 提供训练、测试与预测脚本。

### ModelTraceability

- 通过迭代优化进行可追溯重构分析。
- 支持 `L1/L2/SSIM/PSNR/LPIPS` 等多种距离度量。

### RiskUserMining

- 在图结构数据上执行风险用户挖掘（GraphSAGE 变体）。
- 包含训练、验证、测试及 AUC 评估流程。

## 开源说明

- 当前仓库为研究/原型版本，适合二次开发与学术/工程探索。
- 部分子目录包含第三方代码，请遵循其各自 License 与使用条款。
- 生产环境落地前建议补充数据治理、日志审计与模型监控能力。

## 路线图

- [ ] 提供统一 API 网关，整合全部核心模块
- [ ] 增加标准化溯源报告 JSON Schema
- [ ] 增加 Docker Compose 一键部署
- [ ] 增加可复现实验脚本与 Demo 样例

## 致谢

感谢开源社区及各上游项目提供的算法与工程基础。

## License

请遵循 `FlaskBackend/LICENSE`，并同时遵守各第三方子模块的许可证要求。
