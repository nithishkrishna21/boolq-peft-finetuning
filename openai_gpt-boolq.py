import openai
from openai import OpenAI
from datasets import load_dataset
import random
import json
from dotenv import load_dotenv

import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


dataset = load_dataset("boolq")
train_data = dataset['train']
test_data = dataset['validation']

yes_examples = [x for x in train_data if x['answer']]
no_examples = [x for x in train_data if not x['answer']]
demonstrations = random.sample(yes_examples, 4) + random.sample(no_examples, 4)
random.shuffle(demonstrations)

few_shot_prompt = ""
for d in demonstrations:
    ans = "yes" if d['answer'] else "no"
    few_shot_prompt += f"Passage: {d['passage']}\nQuestion: {d['question']}\nAnswer: {ans}\n\n"

test_samples = random.sample(list(test_data), 30)

# evaluate 
correct, total = 0, 0
client = OpenAI()

for t in test_samples:

    prompt = few_shot_prompt + f"Passage: {t['passage']}\nQuestion: {t['question']}\nAnswer:"

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            }
        ],
        text={
            "format": {
            "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=True
    )

    # print(response.output[0].content[0].text, "\n")

    
    predicted_answer = response.output[0].content[0].text
    ground_truth = "yes" if t["answer"] else "no"
    
    if predicted_answer == ground_truth:
        correct += 1
    total += 1

# Compute accuracy
accuracy = correct / total
print(f"Evaluation Accuracy: {accuracy:.2%}")