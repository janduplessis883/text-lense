import requests
import json


# %%
def ask_ollama(
    user_prompt,
    system_prompt="You are a helpful AI assistant.",
    model="gpt-oss:latest",
    format="plain text",
    temp=0,
    max_tokens=0,
):
    """
    import the following for function to work:
    from helper883 import *
    """

    output_format = (
        f"PLEASE MAKE SURE YOUR OUTPUT IS IN THE FOLLOWING FORMAT: {format}."
    )
    # Define the endpoint URL and headers
    url = "http://localhost:11434/api/generate"  # Update the URL to match your server's endpoint
    headers = {
        "Content-Type": "application/json",
    }

    # Define the payload with the prompt or input text
    payload = {
        "model": model,
        "system": f"system_prompt {output_format}",
        "prompt": user_prompt,
        "max_tokens": max_tokens,
        "stream": False,
        "options": {
            "temperature": temp,
        },
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Check the response
    if response.status_code == 200:
        result = response.json()
        return result["response"]
    else:
        print("Error:", response.status_code, response.text)
        return False
