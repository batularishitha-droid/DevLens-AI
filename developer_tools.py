import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=120
    )

    if response.status_code == 200:
        return response.json()["message"]["content"]

    return f"Ollama error: {response.status_code}"


def analyze_error(error_text):

    prompt = f"""
You are an expert debugging assistant.

Analyze this error:

{error_text}

Give:

## Error
Identify the error.

## Cause
Explain why it happened.

## Solution
Give the solution.

## Corrected Code
Give corrected code if necessary.

## Prevention
Explain how to avoid it.

Use simple language.
"""

    return ask_ollama(prompt)


def analyze_code(code):

    prompt = f"""
You are an expert programming teacher.

Analyze this code:

{code}

Give:

## What the code does
Explain the purpose.

## Explanation
Explain the important parts.

## Bugs
Find possible bugs.

## Improvements
Suggest improvements.

## Improved Code
Give improved code if useful.

Use simple language.
"""

    return ask_ollama(prompt)


def review_code(code):

    prompt = f"""
You are a professional code reviewer.

Review this code:

{code}

Check:

1. Correctness
2. Bugs
3. Readability
4. Performance
5. Security
6. Best practices

Give:

## Code Quality Score
Give a score from 1 to 10.

## Problems
List the problems.

## Suggestions
Give improvements.

## Improved Code
Give improved code when useful.
"""

    return ask_ollama(prompt)