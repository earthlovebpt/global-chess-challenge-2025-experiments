# Player Agents

## Creating Your Chess Agent

To create your own chess agent, you need to provide a **prompt template** that matches the input format expected by your LLM.

### Prompt Template

Your prompt template should be a Jinja2 template file (`.jinja`) that uses the following variables:

- `{{ fen }}` - The current board position in FEN format
- `{{ legal_moves_list }}` - List of legal moves in UCI format
- `{{ board_ascii }}` - ASCII representation of the current board
- `{{ first_legal_move }}` - An example of a valid legal move

### Output Format

Your LLM should output moves in this format:

```
<think>brief reasoning</think>
<uci_move>your_move_here</uci_move>
```

The move must be in UCI format and must be one of the legal moves from `legal_moves_list`.

### Example

See `llm_agent_prompt_template.jinja` for a working example.

