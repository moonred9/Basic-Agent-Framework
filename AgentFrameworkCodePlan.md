# Agent Framework Code Plan

本文档基于 `AgentFrameworkPlan.md` 中的工作流，规划一个可扩展的 Agent 框架代码组织。当前目标不是一次性实现复杂能力，而是先建立清晰的边界：Agent 负责循环，Context 负责生成 LLM messages，Memory 负责管理外部信息，Tool 负责执行动作。

## 1. 核心工作流

框架主流程建议保持如下顺序：

1. Agent 接收用户输入，初始化一次运行会话。
2. Memory 记录用户输入、系统信息、工具结果等可被上下文使用的信息。
3. Context 从 Memory 和运行状态中抽取有效信息，并转换为当前 LLM 底座实际接收的 `messages`。
4. LLM 生成包含 reasoning、tool call 或 answer 的结构化输出。
5. Agent 解析 LLM 输出：
   - 如果调用 `search` tool，则执行检索，将结果写入 Memory，然后进入下一轮。
   - 如果调用 `answer` tool，则输出最终答案，并终止循环。
6. 如果达到最大轮数仍未调用 `answer`，Agent 返回兜底结果或抛出可识别异常。

与原始文档的主要区别：最终回答不直接依赖 `<answer>` 标签，而是通过 `answer` tool 明确终止循环。

## 2. 推荐目录结构

```text
basic_agent_framework/
  __init__.py
  agent/
    __init__.py
    agent.py
    loop.py
    state.py
  context/
    __init__.py
    context.py
    adapters.py
    message.py
  memory/
    __init__.py
    memory.py
    records.py
  llm/
    __init__.py
    client.py
    output_parser.py
    schemas.py
  tools/
    __init__.py
    base.py
    registry.py
    answer.py
    search.py
  runtime/
    __init__.py
    config.py
    errors.py
```

后续可以增加：

```text
tests/
  test_agent_loop.py
  test_context.py
  test_memory.py
  test_tools.py
```

## 3. 核心类规划

### 3.1 Agent

文件：`basic_agent_framework/agent/agent.py`

职责：

- 作为框架对外主入口。
- 接收用户输入并启动一次 agent run。
- 协调 Memory、Context、LLM Client、Tool Registry。
- 管理最大轮数和 answer 终止逻辑。

建议接口：

```python
class Agent:
    def __init__(
        self,
        llm_client: LLMClient,
        context: Context,
        memory: Memory,
        tool_registry: ToolRegistry,
        config: AgentConfig,
    ) -> None:
        ...

    def run(self, user_input: str) -> AgentResult:
        ...
```

关键行为：

- `run()` 内部创建或重置 `AgentState`。
- 每轮调用 `context.build_messages(memory=memory, state=state)`。
- 调用 `llm_client.generate(messages)`。
- 使用 `OutputParser` 将 LLM 输出解析为 `ToolCall`。
- 执行 tool 后将 `ToolResult` 写回 Memory。
- 如果 tool 名称是 `answer`，立即返回 `AgentResult`。

### 3.2 AgentState

文件：`basic_agent_framework/agent/state.py`

职责：

- 保存单次运行中的临时状态。
- 不负责长期记忆。

建议字段：

```python
@dataclass
class AgentState:
    run_id: str
    user_input: str
    current_step: int = 0
    max_steps: int = 10
    finished: bool = False
```

可选字段：

- `last_llm_output: str | None`
- `last_tool_call: ToolCall | None`
- `answer: str | None`

### 3.3 Context

文件：`basic_agent_framework/context/context.py`

职责：

- 专门负责把 Memory 和运行状态转换成 LLM 实际接收的 messages。
- 屏蔽不同 LLM 底座的消息格式差异。
- 处理 reasoning tag、tool call tag、tool response tag、answer tool 指令等格式差异。

外部调用 Context 时只注入有效信息，例如 Memory、AgentState、可用工具说明，而不直接拼 prompt。

建议接口：

```python
class Context:
    def __init__(
        self,
        adapter: ContextAdapter,
        system_prompt: str,
        tool_registry: ToolRegistry,
    ) -> None:
        ...

    def build_messages(self, memory: Memory, state: AgentState) -> list[Message]:
        ...
```

关键行为：

- 从 Memory 中获取用户输入、历史工具调用、工具响应、可用证据。
- 注入系统约束，例如：
  - 必须先 reasoning，再选择 tool。
  - 可用工具只有 `search` 和 `answer`。
  - 调用 `answer` 后循环结束。
- 调用 `ContextAdapter` 将内部统一消息结构转换成具体 LLM messages。

### 3.4 ContextAdapter

文件：`basic_agent_framework/context/adapters.py`

职责：

- 适配不同模型的消息格式和特殊标签。
- 例如某些模型使用 `<think>`，某些模型可能要求隐藏 reasoning 或使用其他 tool call 格式。

建议接口：

```python
class ContextAdapter(Protocol):
    def render_system_message(self, content: str) -> Message:
        ...

    def render_user_message(self, content: str) -> Message:
        ...

    def render_tool_response(self, tool_name: str, content: str) -> Message:
        ...

    def render_tool_instructions(self, tools: list[ToolSpec]) -> str:
        ...
```

默认实现：

```python
class TaggedContextAdapter:
    reasoning_tag = "think"
    tool_call_tag = "tool_call"
    tool_response_tag = "tool_response"
```

默认输出可约定为：

```text
<think>...</think>
<tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>
```

最终回答通过：

```text
<tool_call>{"name": "answer", "arguments": {"answer": "..."}}</tool_call>
```

### 3.5 Message

文件：`basic_agent_framework/context/message.py`

职责：

- 框架内部统一消息结构。
- 避免在各处直接使用不同 provider 的 message dict。

建议结构：

```python
@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 3.6 Memory

文件：`basic_agent_framework/memory/memory.py`

职责：

- 管理所有可注入 Context 的外部信息。
- Memory 不负责决定最终 prompt 格式，只负责保存、筛选和输出信息。
- Memory 输出内容给 Context，Context 再把这些内容编成某个 message。

建议接口：

```python
class Memory:
    def add_user_input(self, content: str) -> None:
        ...

    def add_llm_output(self, content: str) -> None:
        ...

    def add_tool_call(self, tool_call: ToolCall) -> None:
        ...

    def add_tool_result(self, result: ToolResult) -> None:
        ...

    def get_context_records(self, limit: int | None = None) -> list[MemoryRecord]:
        ...

    def clear(self) -> None:
        ...
```

初期可以实现为内存列表：

```python
class InMemoryMemory(Memory):
    records: list[MemoryRecord]
```

后续扩展方向：

- 长期记忆存储。
- 向量检索。
- 按 token budget 裁剪上下文。
- 按 record type 过滤，例如只取工具结果或用户输入。

### 3.7 MemoryRecord

文件：`basic_agent_framework/memory/records.py`

职责：

- 统一记录 Memory 中的事件。

建议结构：

```python
@dataclass
class MemoryRecord:
    id: str
    type: Literal["user_input", "llm_output", "tool_call", "tool_result", "system"]
    content: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 3.8 LLMClient

文件：`basic_agent_framework/llm/client.py`

职责：

- 封装底层 LLM 调用。
- Agent 不直接依赖具体 SDK。

建议接口：

```python
class LLMClient(Protocol):
    def generate(self, messages: list[Message]) -> LLMResponse:
        ...
```

建议结构：

```python
@dataclass
class LLMResponse:
    content: str
    raw: Any | None = None
```

### 3.9 OutputParser

文件：`basic_agent_framework/llm/output_parser.py`

职责：

- 从 LLM 输出中解析 tool call。
- 初期支持 `<tool_call>...</tool_call>` 中的 JSON。
- 后续可支持 provider-native tool call。

建议接口：

```python
class OutputParser:
    def parse(self, response: LLMResponse) -> ParsedOutput:
        ...
```

建议结构：

```python
@dataclass
class ParsedOutput:
    reasoning: str | None
    tool_call: ToolCall | None
    raw_text: str
```

解析规则：

- 允许存在 `<think>...</think>`，但业务逻辑不依赖 reasoning。
- 必须解析出一个 tool call。
- 如果没有 tool call，可以转为 `ParseError`，或者由 Agent 触发一次修复提示。

### 3.10 Tool

文件：`basic_agent_framework/tools/base.py`

职责：

- 定义工具统一接口。
- 所有工具都通过 `ToolCall` 输入，通过 `ToolResult` 输出。

建议接口：

```python
class Tool(Protocol):
    name: str
    description: str

    def spec(self) -> ToolSpec:
        ...

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        ...
```

建议结构：

```python
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]

@dataclass
class ToolResult:
    name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
```

### 3.11 ToolRegistry

文件：`basic_agent_framework/tools/registry.py`

职责：

- 管理工具注册、查找和执行。
- Context 通过 Registry 获取工具说明。
- Agent 通过 Registry 执行工具。

建议接口：

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None:
        ...

    def get(self, name: str) -> Tool:
        ...

    def list_specs(self) -> list[ToolSpec]:
        ...

    def run(self, tool_call: ToolCall) -> ToolResult:
        ...
```

### 3.12 SearchTool

文件：`basic_agent_framework/tools/search.py`

职责：

- 执行检索。
- 当前可以先做一个 stub 或简单本地检索接口，后续再接 FAISS、网络搜索、文档库等。

建议接口：

```python
class SearchTool:
    name = "search"
    description = "Search external knowledge and return relevant evidence."

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments["query"]
        ...
```

参数：

```json
{
  "query": "string"
}
```

返回内容建议包含：

- 查询词。
- 检索到的 chunk 列表。
- 来源 metadata。

### 3.13 AnswerTool

文件：`basic_agent_framework/tools/answer.py`

职责：

- 接收最终答案。
- 调用该 tool 后 Agent 循环终止。
- 工具本身不做复杂逻辑，只把答案标准化为 `ToolResult`。

建议接口：

```python
class AnswerTool:
    name = "answer"
    description = "Return the final answer and stop the agent loop."

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        answer = arguments["answer"]
        ...
```

参数：

```json
{
  "answer": "string"
}
```

Agent 侧规则：

```python
if tool_call.name == "answer":
    state.finished = True
    return AgentResult(answer=tool_result.content, steps=state.current_step)
```

## 4. 数据流

一次 search 循环的数据流：

```text
user_input
  -> Memory.add_user_input()
  -> Context.build_messages()
  -> LLMClient.generate()
  -> OutputParser.parse()
  -> ToolRegistry.run(search)
  -> Memory.add_tool_result()
  -> next step
```

一次 answer 终止的数据流：

```text
Context.build_messages()
  -> LLMClient.generate()
  -> OutputParser.parse()
  -> ToolRegistry.run(answer)
  -> Memory.add_tool_result()
  -> AgentResult
```

## 5. Agent 循环伪代码

```python
def run(self, user_input: str) -> AgentResult:
    state = AgentState(
        run_id=uuid4().hex,
        user_input=user_input,
        max_steps=self.config.max_steps,
    )
    self.memory.add_user_input(user_input)

    while state.current_step < state.max_steps:
        messages = self.context.build_messages(self.memory, state)
        response = self.llm_client.generate(messages)
        self.memory.add_llm_output(response.content)

        parsed = self.output_parser.parse(response)
        if parsed.tool_call is None:
            raise ParseError("LLM output does not contain a tool call.")

        self.memory.add_tool_call(parsed.tool_call)
        tool_result = self.tool_registry.run(parsed.tool_call)
        self.memory.add_tool_result(tool_result)

        state.current_step += 1

        if parsed.tool_call.name == "answer":
            state.finished = True
            state.answer = tool_result.content
            return AgentResult(
                answer=tool_result.content,
                steps=state.current_step,
                finished=True,
            )

    raise MaxStepsExceededError(state.max_steps)
```

## 6. 配置规划

文件：`basic_agent_framework/runtime/config.py`

建议结构：

```python
@dataclass
class AgentConfig:
    max_steps: int = 10
    require_tool_call: bool = True
    stop_tool_name: str = "answer"
```

后续可加入：

- `token_budget`
- `model_name`
- `temperature`
- `enable_reasoning`
- `context_adapter_name`

## 7. 错误类型

文件：`basic_agent_framework/runtime/errors.py`

建议错误：

```python
class AgentFrameworkError(Exception):
    pass

class ToolNotFoundError(AgentFrameworkError):
    pass

class ToolExecutionError(AgentFrameworkError):
    pass

class ParseError(AgentFrameworkError):
    pass

class MaxStepsExceededError(AgentFrameworkError):
    pass
```

## 8. 第一阶段实现顺序

建议按以下顺序实现，降低耦合风险：

1. 定义数据结构：`Message`、`MemoryRecord`、`ToolCall`、`ToolResult`、`ToolSpec`、`AgentState`、`AgentResult`。
2. 实现 `Memory` 和 `InMemoryMemory`。
3. 实现 `Tool`、`ToolRegistry`、`AnswerTool`、`SearchTool`。
4. 实现 `ContextAdapter` 和 `Context`。
5. 实现 `OutputParser`。
6. 实现 `Agent.run()` 主循环。
7. 用 fake LLM client 写单元测试，验证 search 多轮和 answer 终止。

## 9. 最小可用测试场景

### 场景 1：直接回答

LLM 第一次返回：

```text
<think>已有足够信息。</think>
<tool_call>{"name": "answer", "arguments": {"answer": "最终答案"}}</tool_call>
```

期望：

- Agent 执行 `answer`。
- 循环立即终止。
- `AgentResult.finished == True`。
- `AgentResult.answer == "最终答案"`。

### 场景 2：先搜索再回答

LLM 第一次返回：

```text
<think>需要检索。</think>
<tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>
```

LLM 第二次返回：

```text
<think>已有证据。</think>
<tool_call>{"name": "answer", "arguments": {"answer": "基于检索结果的答案"}}</tool_call>
```

期望：

- 第一轮执行 `search`。
- search 结果进入 Memory。
- 第二轮 Context messages 包含 search 结果。
- 第二轮执行 `answer` 并终止。

## 10. 设计原则

- Agent 只编排流程，不拼 prompt，不理解具体工具细节。
- Context 只负责构建 messages，不保存长期状态。
- Memory 只负责存储和筛选信息，不关心 LLM 格式。
- Tool 只负责执行动作，不控制循环。
- `answer` 是一个普通 tool，但 Agent 对它有特殊终止规则。
- LLM 输出解析集中在 `OutputParser`，避免散落在 Agent 循环中。


# 将 GPU1 改为 GPU2 (或者任何不重复的字符串)
CUDA_VISIBLE_DEVICES=3 \
VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME=VLLM_OBJECT_STORAGE_SHM_BUFFER_GPU3 \
vllm serve /data/pjj/llm/GLM-4.6V-Flash \
  --served-model-name glm-4.6v \
  --tensor-parallel-size 1 \
  --tool-call-parser glm45 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --allowed-local-media-path / \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --host 10.130.140.30 \
  --port 18804