### 一、SFT（监督微调，Supervised Fine-Tuning）

#### 核心目标

让预训练模型（仅具备通用语言能力）学会理解人类指令并生成符合预期的回应，实现“基础指令对齐”，是所有后续对齐技术的基础。

#### 核心原理

通过**监督学习**，最小化模型对“人类标注的目标回应”的预测损失，使模型在给定指令时，生成与标注内容高度相似的输出。

- 损失函数：交叉熵损失（Cross-Entropy Loss）
  LSFT=−1N∑i=1Nlog⁡(p(yi∣x,θ))\mathcal{L}_{\text{SFT}} = -\frac{1}{N} \sum_{i=1}^N \log(p(y_i | x, \theta))LSFT​=−N1​i=1∑N​log(p(yi​∣x,θ))
  其中，xxx为指令，yiy_iyi​为标注的第iii个token，p(yi∣x,θ)p(y_i | x, \theta)p(yi​∣x,θ)为模型预测的token概率，θ\thetaθ为模型参数。

#### 优缺点

| 优点                                                         | 缺点                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1. 逻辑简单，易实现（无需复杂强化学习组件）； 2. 是DPO/PPO/GRPO的基础，无SFT则后续对齐无效； 3. 对简单指令（如问答、格式生成）效果稳定。 | 1. 依赖大量高质量人工标注数据（成本高，10k样本标注成本约数万元）； 2. 泛化性差（未见过的指令可能输出偏离预期内容）； 3. 存在“暴露偏差”（训练时见完整标注，推理时逐token生成，易累积错误）。 |

#### 实践过程

##### 输入

- 数据格式

  ：“指令-回应”配对数据（JSON格式）

  ```json
  [
  {"instruction": "写一封请假邮件", "input": "", "output": "尊敬的领导：因身体不适，需请假1天，望批准。"}
  ]
  ```

- **数据规模**：10k100k条（中小模型），100k1M条（大模型）；

- **数据来源**：公开数据集（Alpaca、ShareGPT）、人工标注、大模型生成（数据蒸馏）。

##### 输出

- 具备基础指令遵循能力的模型（如Llama-3-8B-SFT、Qwen-1.5-7B-SFT）；
- 模型 checkpoint 文件（含权重、配置、tokenizer）。

##### 可执行方案（基于Hugging Face生态）

1. **数据准备与清洗**

   ```python
   import json
   # 加载原始数据  
   with open("sft_data.json", "r") as f:
   data = json.load(f)
   # 清洗：过滤长度>500token、含有害词的样本  
   clean_data = []
   for item in data:
   if len(item["output"]) < 500 and "有害词" not in item["output"]:
   clean_data.append(item)
   # 保存清洗后数据  
   with open("clean_sft_data.json", "w") as f:
   json.dump(clean_data, f, indent=2)
   ```

2. **模型与tokenizer加载**

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model_name = "meta-llama/Llama-3-8B"  # 预训练模型  
   model = AutoModelForCausalLM.from_pretrained(
   model_name,
   device_map="auto",  # 自动分配GPU/CPU  
   torch_dtype="auto"  # 自动选择数据类型（如BF16）  
   )
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   tokenizer.pad_token = tokenizer.eos_token  # 设置pad token（必填）  
   ```

3. **数据集格式化（转为模型输入格式）**

   ```python
   def format_prompt(item):
   return f"[INST] {item['instruction']} [/INST] {item['output']}"
   # 转换为tokenized数据集  
   from datasets import Dataset
   dataset = Dataset.from_list(clean_data)
   def tokenize_function(examples):
   prompts = [format_prompt(item) for item in examples]
   return tokenizer(prompts, truncation=True, max_length=512, padding="max_length")
   tokenized_dataset = dataset.map(tokenize_function, batched=True)
   ```

4. **训练配置与启动**

   ```python
   from transformers import TrainingArguments, Trainer
   training_args = TrainingArguments(
   output_dir="./llama3-8b-sft",  # 模型保存路径  
   per_device_train_batch_size=8,  # 单卡batch size（12GB显存建议8）  
   learning_rate=3e-5,  # 学习率（中小模型常用3e-5）  
   num_train_epochs=3,  # 训练轮数（避免过拟合）  
   logging_steps=100,  # 每100步打印日志  
   save_strategy="epoch",  # 每轮保存一次模型  
   fp16=True,  # 混合精度训练（加速且省显存）  
   report_to="none"  # 不使用wandb等监控工具  
   )
   trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset
   )
   trainer.train()  # 启动训练  
   ```

##### 注意事项

1. **数据质量优先**：10k高质量样本（无错误、逻辑清晰）优于100k低质样本；
2. **避免过拟合**：验证集损失上升时停止训练（设置`load_best_model_at_end=True`）；
3. **学习率选择**：模型越大，学习率越小（如70B模型用1e-5，7B模型用3e-5）；
4. **保留预训练能力**：冻结底层20%权重（可选），防止模型遗忘通用知识。

### 二、DPO（直接偏好优化，Direct Preference Optimization）

#### 核心目标

跳过奖励模型（RM）训练，直接用“人类偏好对”（优质回应vs劣质回应）优化模型，实现“高效偏好对齐”，降低PPO的复杂性。

#### 核心原理

通过**偏好对比损失**，让模型对“优质回应”的生成概率显著高于“劣质回应”，无需显式奖励信号。

- 损失函数：
  LDPO=−log⁡(σ(β⋅(log⁡pθ(chosen∣x)pθ(rejected∣x)−log⁡pref(chosen∣x)pref(rejected∣x))))\mathcal{L}_{\text{DPO}} = -\log\left( \sigma\left( \beta \cdot \left( \log\frac{p_\theta(\text{chosen}|x)}{p_\theta(\text{rejected}|x)} - \log\frac{p_{\text{ref}}(\text{chosen}|x)}{p_{\text{ref}}(\text{rejected}|x)} \right) \right) \right)LDPO​=−log(σ(β⋅(logpθ​(rejected∣x)pθ​(chosen∣x)​−logpref​(rejected∣x)pref​(chosen∣x)​)))
  其中，pθp_\thetapθ​为待优化模型，prefp_{\text{ref}}pref​为参考模型（SFT模型），β\betaβ为温度参数（控制偏好强度），chosen\text{chosen}chosen为优质回应，rejected\text{rejected}rejected为劣质回应。

#### 优缺点

| 优点                                                         | 缺点                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1. 无需训练奖励模型（节省50%计算成本）； 2. 数据效率高（仅需SFT数据量的1/10）； 3. 训练稳定（依赖参考模型约束，无模式崩溃风险）。 | 1. 依赖高质量偏好对（优质/劣质差异需明确）； 2. 对复杂偏好（如多轮对话连贯性）效果弱于PPO； 3. 参考模型质量决定上限（SFT差则DPO无效）。 |

#### 实践过程

##### 输入

- 数据格式

  ：“指令+优质回应+劣质回应”三元组

  ```json
  [
  {
  "instruction": "推荐一本科幻小说",
  "chosen": "《三体》：刘慈欣的经典，探讨宇宙文明法则",  # 优质
  "rejected": "《西游记》：不是科幻小说"  # 劣质
  }
  ]
  ```

- **数据规模**：1k~10k条（远少于SFT）；

- **数据来源**：人工标注偏好对、SFT模型生成多候选后筛选。

##### 输出

- 符合人类偏好的模型（如Llama-3-8B-DPO）；
- 模型checkpoint（相比SFT，更倾向生成“优质回应”风格的内容）。

##### 可执行方案（基于TRL库）

1. **数据准备**

   ```python
   # 加载偏好数据  
   with open("dpo_data.json", "r") as f:
   dpo_data = json.load(f)
   # 转换为TRL库要求的格式（需包含"chosen"和"rejected"字段）  
   from datasets import Dataset
   dpo_dataset = Dataset.from_list(dpo_data)
   ```

2. **模型初始化**

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from trl import DPOTrainer, DPOConfig
   # 参考模型（固定SFT模型）  
   ref_model = AutoModelForCausalLM.from_pretrained("./llama3-8b-sft", device_map="auto")
   # 待优化模型（复制SFT权重）  
   model = AutoModelForCausalLM.from_pretrained("./llama3-8b-sft", device_map="auto")
   tokenizer = AutoTokenizer.from_pretrained("./llama3-8b-sft")
   ```

3. **DPO训练配置与启动**

   ```python
   dpo_config = DPOConfig(
   output_dir="./llama3-8b-dpo",
   per_device_train_batch_size=16,  # 偏好数据少，可设大batch  
   learning_rate=2e-5,  # 小于SFT，避免破坏基础能力  
   num_train_epochs=2,  # 偏好数据易过拟合  
   beta=0.1,  # 温度参数（常用0.1~0.5）  
   logging_steps=50
   )
   # 初始化DPO Trainer  
   dpo_trainer = DPOTrainer(
   model=model,
   ref_model=ref_model,
   args=dpo_config,
   train_dataset=dpo_dataset,
   tokenizer=tokenizer,
   max_prompt_length=256,  # 指令最大长度  
   max_length=512  # 回应最大长度  
   )
   dpo_trainer.train()  # 启动训练  
   ```

##### 注意事项

1. **偏好对质量**：避免“模糊对”（优质/劣质差异不明显），否则模型无法学习偏好；
2. **β值调整**：β过小（如0.05）会导致偏好学习不充分；β过大（如1.0）会导致模型输出保守（只说安全但无用的内容）；
3. **参考模型固定**：训练中ref_model权重不可更新，否则失去约束作用；
4. **数据增强**：对稀缺偏好类型（如“代码简洁性”），用大模型生成相似样本扩充。

### 三、PPO（近端策略优化，Proximal Policy Optimization）

#### 核心目标

通过**强化学习（RLHF）**，让模型在“奖励模型（RM）打分”引导下优化输出，实现“复杂偏好对齐”（如多维度加权：准确+安全+简洁）。

#### 核心原理

将模型视为“策略网络”，指令视为“环境”，RM打分为“奖励”，通过“采样-打分-更新”循环优化策略，同时用“裁剪损失”限制更新幅度，平衡“奖励提升”与“策略稳定”。

- 裁剪损失函数：
  LPPO=min⁡(rt⋅At,clip(rt,1−ϵ,1+ϵ)⋅At)\mathcal{L}_{\text{PPO}} = \min\left( r_t \cdot A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t \right)LPPO​=min(rt​⋅At​,clip(rt​,1−ϵ,1+ϵ)⋅At​)
  其中，rtr_trt​为当前策略与旧策略的概率比，AtA_tAt​为优势值（实际奖励-预期奖励），ϵ\epsilonϵ为裁剪系数（控制更新幅度）。

#### 优缺点

| 优点                                                         | 缺点                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1. 支持多维度偏好（如安全分*0.3+准确分*0.7）； 2. 泛化性强（对未见过的指令，能通过奖励调整输出）； 3. 可控性高（可通过奖励函数引导输出方向）。 | 1. 流程复杂（需训SFT、RM、策略网络、价值网络）； 2. 计算成本高（是DPO的3~5倍）； 3. 易出现“模式崩溃”（模型只生成高奖励但无意义内容）。 |

#### 实践过程

##### 输入

1. **SFT模型**：作为策略网络初始化权重；
2. **奖励模型（RM）训练数据**：“指令+回应+人类打分（1-10分）”三元组；
3. **策略更新数据**：大规模指令集（用于生成候选回应）。

##### 输出

- 符合复杂偏好的强化学习模型（如Llama-3-8B-PPO）；
- 模型checkpoint（相比DPO，更能平衡多维度偏好）。

##### 可执行方案（基于TRL库）

1. **训练奖励模型（RM）**

   ```python
   # 定义奖励模型（基于SFT模型添加奖励头）  
   from transformers import AutoModelForCausalLM, AutoTokenizer
   import torch.nn as nn
   class RewardModel(nn.Module):
   def __init__(self, base_model):
   super().__init__()
   self.base_model = base_model
   self.reward_head = nn.Linear(base_model.config.hidden_size, 1)  # 输出标量奖励  
   def forward(self, input_ids, attention_mask):
   outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
   last_hidden = outputs.last_hidden_state[:, -1, :]  # 取最后token的隐藏态  
   return self.reward_head(last_hidden).squeeze(-1)  # 输出奖励分  
   # 加载SFT模型作为基础  
   base_model = AutoModelForCausalLM.from_pretrained("./llama3-8b-sft")
   reward_model = RewardModel(base_model)
   # 训练RM（简化代码，实际需用MSE损失训练）  
   ```

2. **PPO策略优化**

   ```python
   from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
   # 策略网络（带价值头，用于估计预期奖励）  
   model = AutoModelForCausalLMWithValueHead.from_pretrained("./llama3-8b-sft")
   # 配置PPO  
   ppo_config = PPOConfig(
   output_dir="./llama3-8b-ppo",
   learning_rate=5e-6,  # 策略更新需谨慎，用小学习率  
   batch_size=32,
   eps_clip=0.2,  # 裁剪系数（0.1~0.2）  
   kl_coef=0.05,  # KL惩罚（防止策略偏移）  
   num_train_epochs=3
   )
   # 加载指令数据集（用于生成回应）  
   instruction_dataset = Dataset.from_list([{"instruction": "..."}, ...])
   # 初始化PPO Trainer  
   ppo_trainer = PPOTrainer(
   model=model,
   config=ppo_config,
   tokenizer=tokenizer,
   dataset=instruction_dataset,
   reward_model=reward_model  # 奖励模型  
   )
   # 迭代训练（采样-打分-更新）  
   for epoch in range(3):
   # 1. 生成回应（采样）  
   outputs = ppo_trainer.generate(batch_size=128, max_new_tokens=200)
   # 2. 奖励打分  
   rewards = reward_model(** outputs["response_ids"])
   # 3. 更新策略  
   stats = ppo_trainer.train(minibatch_size=32)
   ```

##### 注意事项

1. **奖励模型设计**：多维度奖励需合理加权（如安全分权重过高会导致模型“不说有用内容”）；
2. **控制KL散度**：策略与SFT模型的KL值需保持在0.01~0.05（过高说明策略偏移，过低说明无提升）；
3. **避免奖励欺骗**：定期人工检查模型输出，防止模型生成“看似高分实则无意义”的内容（如重复关键词）；
4. **混合SFT数据**：每轮训练加入10%的SFT数据，防止策略遗忘基础指令能力。

### 四、GRPO（组相对策略优化，Group Relative Policy Optimization）

#### 核心目标

改进PPO，用“组内相对奖励”替代价值网络，降低计算成本，提升**推理密集型任务**（数学、代码）的对齐效果。

#### 核心原理

对每个指令生成“一组候选回应”，通过组内奖励对比计算“相对优势”（无需价值网络），结合KL约束优化策略，增强训练效率与稳定性。

- 相对优势计算：
  r~i=ri−mean(r)std(r)+1e−8\tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + 1e-8}r~i​=std(r)+1e−8ri​−mean(r)​
  其中，r\mathbf{r}r为组内所有奖励，r~i\tilde{r}_ir~i​为第iii个回应的相对优势（归一化后更抗噪声）。

#### 优缺点

| 优点                                                         | 缺点                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1. 无需价值网络（内存占用减少50%）； 2. 训练速度比PPO快30%； 3. 组内对比适合推理任务（如数学多解法评估）。 | 1. 采样成本高（需生成多候选回应）； 2. 依赖组内奖励分布（分布不均则效果差）； 3. 对简单任务（如短对话）优势不明显。 |

#### 实践过程

##### 输入

1. **SFT模型**：作为策略网络初始化；
2. **组采样数据**：对每个指令生成GGG个候选回应（G=8∼32G=8\sim32G=8∼32，推理任务用大GGG）；
3. **奖励模型**：同PPO（支持多维度打分）。

##### 输出

- 强推理能力的对齐模型（如Llama-3-8B-GRPO）；
- 模型checkpoint（在数学、代码任务上表现优于PPO）。

##### 可执行方案（基于TRL库）

1. **组采样数据生成**

   ```python
   def generate_group(instruction, model, tokenizer, G=16):
   """为单个指令生成G个候选回应"""
   inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
   responses = []
   for _ in range(G):
   # 带随机性生成，确保候选多样性  
   outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.8, do_sample=True)
   responses.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
   return {"instruction": instruction, "responses": responses}
   # 生成组数据（示例）  
   group_data = [generate_group(inst["instruction"], model, tokenizer) for inst in instruction_dataset]
   ```

2. **GRPO训练配置与启动**

   ```python
   from trl import GRPOTrainer, GRPOConfig
   grpo_config = GRPOConfig(
   output_dir="./llama3-8b-grpo",
   learning_rate=5e-6,
   group_size=16,  # 组内候选数（推理任务用32）  
   kl_coef=0.03,  # KL约束（比PPO更严格）  
   num_train_epochs=2
   )
   # 初始化GRPO Trainer  
   grpo_trainer = GRPOTrainer(
   model=model,
   config=grpo_config,
   tokenizer=tokenizer,
   train_dataset=group_data,  # 组采样数据  
   reward_model=reward_model
   )
   grpo_trainer.train()  # 启动训练  
   ```

##### 注意事项

1. **组大小选择**：数学/代码任务用G=32G=32G=32（提升对比效果），简单任务用G=8G=8G=8（降低成本）；
2. **异步采样**：用单独进程生成组数据，避免训练中断（可节省50%时间）；
3. **奖励归一化**：组内奖励必须标准化（否则相对优势计算失效）；
4. **拒绝采样**：每轮筛选组内Top 20%高奖励回应，作为下一轮训练的补充数据（提升效率）。

### 总结：方法选择与组合策略

| 场景需求                              | 推荐方法 | 典型组合       |
| ------------------------------------- | -------- | -------------- |
| 基础指令对齐（如问答机器人）          | SFT      | -              |
| 快速偏好对齐（资源有限）              | DPO      | SFT → DPO      |
| 复杂偏好（如客服机器人，需安全+准确） | PPO      | SFT → RM → PPO |
| 推理/代码任务（如数学解题、代码生成） | GRPO     | SFT → GRPO     |

**核心原则**：所有方法均以SFT为基础，数据质量决定上限，需根据资源与任务复杂度选择组合策略。