# 🤖 data-analyst-agent

A standalone AI agent with code execution capabilities, built with LangGraph and Streamlit.

## Features

- **ReAct-style agent loop** powered by LangGraph
- **Python code execution** in local environment (with persistent namespace between calls)
- **Multi-LLM support**: OpenAI (GPT-4o, GPT-4o-mini) and Anthropic (Claude 3.5 Sonnet, Haiku)
- **File upload**: CSV and Excel file analysis via pandas
- **Financial analysis**: yfinance stock data + agent-generated technical analysis
- **Streaming UI**: Token streaming with syntax-highlighted code display

## Quick Start

### 1. Install dependencies

```bash
cd data-analyst-agent
uv sync
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run the app

```bash
streamlit run app.py
```

Or use the launch script:
```bash
chmod +x run.sh
./run.sh
```

## Environment Variables

See `.env.example` for all available configuration options. Required:
- `OPENAI_API_KEY` - for OpenAI models
- `ANTHROPIC_API_KEY` - for Anthropic/Claude models

## Use Cases

### 1. Data Analysis (CSV/Excel)
1. Upload a CSV or Excel file in the sidebar
2. Ask: *"What are the key statistics in this dataset?"*
3. The agent will write pandas code, execute it, and interpret the results

### 2. Financial Analysis (yfinance)
1. Ask: *"Analyze AAPL stock over the past 6 months with technical indicators"*
2. The agent will fetch data via yfinance, calculate RSI/MACD/Bollinger Bands, and explain the signals

## Project Structure

```
data-analyst-agent/
├── app.py                        # Entry point (streamlit run app.py)
├── requirements.txt              # Python dependencies
├── .env                          # Your API keys (not committed)
├── .env.example                  # Template
├── run.sh                        # Launch script
├── memory-bank/                  # Project documentation
│   ├── projectbrief.md
│   ├── productContext.md
│   ├── systemPatterns.md
│   ├── techContext.md
│   ├── activeContext.md
│   └── progress.md
├── src/
│   ├── agent/
│   │   ├── state.py              # LangGraph AgentState TypedDict
│   │   ├── llm_manager.py        # Multi-LLM factory (OpenAI/Anthropic)
│   │   └── graph.py              # LangGraph agent graph
│   ├── tools/
│   │   └── code_executor.py      # Python REPL tool
│   └── ui/
│       └── streamlit_app.py      # Streamlit UI
└── uploads/                      # Uploaded files (auto-created)
```

## Architecture

```
User Input (Streamlit)
        ↓
  LangGraph Graph
        ↓
  agent_node (LLM)  ←──────────────┐
        ↓                          │
  should_continue?                 │
    ├── YES (tool_calls) → tool_node (python_repl) ──┘
    └── NO → END → Display in UI
```

The agent runs in a ReAct loop:
1. LLM receives user message + conversation history
2. LLM decides to call `python_repl` tool with Python code
3. Code executes locally; output returned to LLM
4. LLM interprets output and either runs more code or gives final answer
5. Variables persist between code calls within the same session

## Code Execution Notes

- File access is **scoped to `data-analyst-agent/`** directory
- Variables defined in one code call **persist** for subsequent calls
- `PROJECT_ROOT` and `UPLOADS_DIR` are pre-injected as variables
- All 3rd-party libraries (pandas, yfinance, etc.) are available

## Token Tracking & Logging

### Token Usage Display
- **Per-turn tokens**: Displayed in the "Execution Output" expander with format:
  ```
  Tokens: Input: X, Input (Agg): Y, Output: Z, Output (Agg): W
  ```
- **Session totals**: Shown in sidebar under "Token Usage (Session)"
- Token counters reset when you click "Clear Conversation"

### LLM Interaction Logs
All LLM interactions are automatically logged to:
```
data-analyst-agent/logs/llm_interactions_{YYYYMMDD}.log
```

Each log entry is in JSONL format (one JSON object per line) containing:
- **Timestamp**: ISO format datetime
- **Provider & Model**: Which LLM was used
- **Message content**: Full text of LLM responses
- **Token usage**: Input/output token counts
- **Metadata**: Response metadata from the LLM provider

Example log entry:
```json
{
  "timestamp": "2026-02-20T14:30:45.123456",
  "provider": "openai",
  "model": "gpt-4o",
  "message_type": "AIMessageChunk",
  "content": "I'll analyze the data...",
  "usage_metadata": {"input_tokens": 150, "output_tokens": 75},
  "response_metadata": {...}
}
```

**Location**: `data-analyst-agent/logs/`

Logs are organized by date - each day gets its own file. This allows you to:
- Track token usage over time
- Debug LLM responses
- Audit conversations
- Analyze agent behavior patterns
