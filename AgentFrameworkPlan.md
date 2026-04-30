

## 大致思路
1. 思考 (Reasoning)： 智能体首先在 <think> 标签内生成推理链，列出当前的认知状态并规划下一步需要检索的信息。
2. 工具调用 (Search/Tool Call)： 智能体通过 <tool_call> 发出一个自然语言查询（Query）。论文提到模型使用的是基于 FAISS 的检索器，并在 2018 年的维基百科数据（wiki-18-corpus）中查找。
3. 环境响应 (Environment Response)： 系统返回检索到的文档chunk，并以 <tool_response> 的形式反馈给模型。
4. 多轮迭代 (Iterative Interaction)： 重复上述过程，将新获取的证据整合进上下文。模型被设置为最高支持 10 轮 交互。
5. 最终回答 (Final Answer)： 当智能体认为已收集足够信息后，在 <answer> 标签中给出简洁的最终答案。