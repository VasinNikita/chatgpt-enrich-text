import os
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("MODEL")
scale_rate = float(os.getenv("SCALE_RATE"))
deviation = (scale_rate - 1) / 4 + 1

input_file = os.getenv("INPUT_FILE")
output_file = os.getenv("OUTPUT_FILE")
prompt_file = os.getenv("PROMPT_FILE")

input_text = open(input_file, "r").read()
prompt_text = open(prompt_file, "r").read()

max_concurrent_workers = 10
max_tokens = 2048
one_word_tokens = 6
max_input_tokens = max_tokens / (scale_rate * 2)

semaphore = asyncio.Semaphore(max_concurrent_workers)
separator = "\n\n"
dot = ". "


def split_string(string, n, delimiter=" "):
    parts = string.split(delimiter)
    filtered_parts = []
    current_part = ""

    for part in parts:
        if len(current_part) + len(part) + 1 <= n:
            current_part += delimiter + part if current_part else part
        else:
            if current_part:
                filtered_parts.append(current_part)
            current_part = part

    if current_part:
        filtered_parts.append(current_part)

    return filtered_parts


def preprocess_text(text):
    paragraphs = split_string(text, max_input_tokens, separator)
    parts = [split_string(paragraph, max_input_tokens, ".") for paragraph in paragraphs]
    return parts


async def openai_request(number, index, buffer):
    async with semaphore:
        async with aiohttp.ClientSession() as session:
            target_buffer = round(len(buffer) * scale_rate)
            target_words = len(buffer.split())
            print(f"({number}.{index}) Waiting for a response... ({len(buffer)} → {target_buffer})\n"
                  f"input_words={len(buffer.split())}\n")

            prompt = (
                f"{prompt_text}{separator}"
                f"The amount of tokens in your response must be at least {round(target_words * 1.1)} but no more than {round(target_words * scale_rate * 1.1)}.{separator}"
                f"{buffer}\n"
            )

            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            }
            data = {
                'model': model,
                "messages": [{"role": "user", "content": prompt}],
                'temperature': 0.2,
                'max_tokens': max_tokens,
                'top_p': 0.5,
                'frequency_penalty': 0,
                'presence_penalty': 0
            }
            async with session.post('https://api.openai.com/v1/chat/completions', headers=headers,
                                    json=data) as response:
                response = await response.json()
                try:
                    enriched_text = response['choices'][0]['message']['content']
                except KeyError:
                    print(response)
                print(
                    f"({number}.{index}) Received a response with ({len(buffer)} → {len(enriched_text)})")
                print(f"output_words={len(enriched_text.split())}\n")
                return number * 1000 + index, enriched_text


async def main(text):
    tasks = []

    for p, paragraph in enumerate(text):
        for s, sentence in enumerate(paragraph):
            tasks.append(asyncio.create_task(openai_request(p + 1, s + 1, sentence)))

    results = await asyncio.gather(*tasks)
    print(results)
    sorted_data = sorted(results, key=lambda x: x[0])

    text = ""
    for x in sorted_data:
        text += x[1]
        text += separator if x[0] % 1000 == 1 else dot

    text = text.replace("..", ".")
    open("output.txt", "w").write(text)
    print(f"\nExpected to get: {round(len(input_text) * scale_rate)}\nGot: {len(text)}")
    print("Done")


asyncio.run(main(preprocess_text(input_text)))
