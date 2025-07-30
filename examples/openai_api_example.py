#!/usr/bin/env python3
"""
Example demonstrating how the vLLM client now uses OpenAI API internally.

This shows the equivalent of what happens inside VLLMClient when you call generate().
"""

from openai import OpenAI

# This is exactly how VLLMClient connects to vLLM internally
client = OpenAI(
    api_key="EMPTY",  # vLLM doesn't require a real API key
    base_url="http://localhost:8000/v1",  # vLLM server with /v1 endpoint
)

# Example 1: Basic generation
print("=== Basic Generation ===")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # vLLM will use whatever model is loaded
    messages=[
        {"role": "user", "content": "What is machine learning?"},
    ],
    max_tokens=512,
    temperature=0.7,
)
print("Response:", response.choices[0].message.content)

# Example 2: With thinking mode enabled (for reasoning models)
print("\n=== With Thinking Mode ===")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Solve: 2x + 5 = 15"},
    ],
    max_tokens=2048,
    temperature=0.7,
    extra_body={
        "top_k": 20,  # vLLM-specific parameter
        "chat_template_kwargs": {
            "enable_thinking": True  # Enable reasoning mode
        }
    }
)
print("Response:", response.choices[0].message.content)

# Example 3: Multiple samples with different temperatures
print("\n=== Multiple Samples ===")
temperatures = [0.5, 0.8, 1.0]
for i, temp in enumerate(temperatures):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Write a creative story opening."},
        ],
        max_tokens=256,
        temperature=temp,
    )
    print(f"Sample {i+1} (temp={temp}):", response.choices[0].message.content[:100] + "...")