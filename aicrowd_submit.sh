#!/bin/bash

# aicrowd login

# Configuration variables
CHALLENGE="global-chess-challenge-2025"
HF_REPO="Qwen/Qwen3-0.6B"
HF_REPO_TAG="main"
PROMPT_TEMPLATE="player_agents/llm_agent_prompt_template.jinja"

echo "Submitting $HF_REPO, if the repo is private, make sure to add access to aicrowd."
echo "Details can can be found in docs/huggingface-gated-models.md"

# Submit the model
aicrowd submit-model \
    --challenge "$CHALLENGE" \
    --hf-repo "$HF_REPO" \
    --hf-repo-tag "$HF_REPO_TAG" \
    --prompt_template_path "$PROMPT_TEMPLATE"