import json
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import requests

load_dotenv()


@dataclass
class MultiplicationTask:
    """Represents a single multiplication task"""
    x: int
    y: int
    mode: str  # "with_prompt", "cot", or "simple"
    task_id: int


@dataclass
class TaskResult:
    """Represents the result of a multiplication task"""
    task: MultiplicationTask
    response_text: str
    extracted_result: int
    expected_result: int
    is_correct: bool
    error: str = None


def extract_multiplication_result(response_text: str) -> Optional[int]:
    """
    Robust extraction of multiplication result from various response formats.
    """
    
    # Strategy 1: Look for explicit RESULT markers (case insensitive)
    result_patterns = [
        r'\*\*RESULT:\s*([0-9,]+)\*\*',  # **RESULT: 123456**
        r'RESULT:\s*([0-9,]+)',          # RESULT: 123456
        r'Result:\s*([0-9,]+)',          # Result: 123456
        r'result:\s*([0-9,]+)',          # result: 123456
        r'Answer:\s*([0-9,]+)',          # Answer: 123456
        r'answer:\s*([0-9,]+)',          # answer: 123456
    ]
    
    for pattern in result_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            number_str = match.group(1).replace(',', '')
            try:
                return int(number_str)
            except ValueError:
                continue
    
    # Strategy 2: Look for multiplication equations (a × b = result)
    multiplication_patterns = [
        r'(\d{1,3}(?:,\d{3})*)\s*[×x*]\s*(\d{1,3}(?:,\d{3})*)\s*=\s*([0-9,]+)',
        r'(\d+)\s*[×x*]\s*(\d+)\s*=\s*([0-9,]+)',
    ]
    
    for pattern in multiplication_patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            # Take the last match (most likely to be the final result)
            last_match = matches[-1]
            result_str = last_match[2].replace(',', '')
            try:
                return int(result_str)
            except ValueError:
                continue
    
    # Strategy 3: Look for "The answer/result/product is X" patterns
    answer_patterns = [
        r'(?:the\s+)?(?:answer|result|product)\s+is\s*:?\s*([0-9,]+)',
        r'(?:answer|result|product):\s*([0-9,]+)',
        r'equals?\s+([0-9,]+)',
        r'is\s+([0-9,]+)\.?\s*$',  # Ends with "is 123456."
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            number_str = match.group(1).replace(',', '')
            try:
                return int(number_str)
            except ValueError:
                continue
    
    # Strategy 4: Extract all large numbers and use heuristics
    # Find all numbers with 10+ digits (likely results of large multiplications)
    large_numbers = re.findall(r'\b(\d{10,})\b', response_text)
    
    if large_numbers:
        # Convert to integers and return the largest (most likely the result)
        try:
            numbers = [int(num) for num in large_numbers]
            return max(numbers)
        except ValueError:
            pass
    
    # Strategy 5: Find all numbers and use context clues
    all_numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*|\d+)\b', response_text)
    
    if all_numbers:
        # Clean and convert numbers
        cleaned_numbers = []
        for num_str in all_numbers:
            try:
                cleaned_num = int(num_str.replace(',', ''))
                cleaned_numbers.append(cleaned_num)
            except ValueError:
                continue
        
        if cleaned_numbers:
            # Heuristic: If we have exactly 3 numbers, the largest is likely the result
            if len(cleaned_numbers) == 3:
                return max(cleaned_numbers)
            
            # Otherwise, return the largest number that's significantly bigger than others
            sorted_nums = sorted(cleaned_numbers, reverse=True)
            if len(sorted_nums) >= 2:
                largest = sorted_nums[0]
                second_largest = sorted_nums[1]
                
                # If the largest is at least 10x bigger than the second largest,
                # it's likely the multiplication result
                if largest >= second_largest * 10:
                    return largest
            
            # Fallback: return the largest number
            return max(cleaned_numbers)
    
    return None

class SimpleRateLimiter:
    """Simple rate limiter with fixed delays"""
    
    def __init__(self, delay_between_requests: float = 1.0):
        self.delay = delay_between_requests
        self.last_request_time = 0
        self.lock = threading.Lock()
        
    def acquire(self):
        """Acquire permission to make a request (blocking)"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.delay:
                sleep_time = self.delay - time_since_last
                time.sleep(sleep_time)
            self.last_request_time = time.time()


class SimpleOpenAIClient:
    """Simple OpenAI client with rate limiting"""
    
    def __init__(self, api_key: str, rate_limiter: SimpleRateLimiter):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.base_url = "https://api.openai.com/v1"
        
    def chat_completion(self, messages: List[Dict], model: str = "gpt-4.1-mini", **kwargs) -> str:
        """Make a chat completion request"""
        self.rate_limiter.acquire()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 1),
            "max_tokens": kwargs.get("max_tokens", 10000),
            "top_p": kwargs.get("top_p", 1),
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=500  # Longer timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")


# Simplified system prompts
system_prompt_with_algorithm = """
Your goal is to perform multiplication of two integers. 

Here’s a memory-lean way to multiply arbitrarily long integers when you can only do single-digit × single-digit and addition. It’s the “column (Comba) long multiplication” algorithm—compute one output column at a time from right to left.


# How the procedure works (step-by-step)

1. Store digits LSB-first
Read each input right-to-left into arrays:
a[0] = ones of A, …, a[n-1] = least-significant; same for b[0..m-1].

2. Initialize
carry = 0; outputs = [] (these will be most-significant to least-significant).

3. Process each result column k = 0 .. n+m-2
Each column k collects all products a[i]*b[j] where i+j = k.

- Start the column sum with the incoming carry: S = carry.
- Sweep i and j=k-i and add each valid single-digit product to S.
- Convert S into:

-- digit_k = S % 10 (the column’s output digit),
-- carry = ⌊S / 10⌋ (what rolls to the next column).

If you don’t have % or /, repeatedly subtract 10 from S while counting how many times (that count is carry; the remainder is digit_k).

- Prepend digit_k to outputs (you’re building MSB→LSB).

4. Flush the carry
After the last column, while carry > 0, keep extracting base-10 digits from carry the same way and prepend them to outputs.

5. Present the result
outputs is MSB→LSB.

# Reusable verbose per-k template (with explicit “skip” lines)

Use this to log every (i, j), including out-of-range skips.
```template
k = {k}:
S = carry = {carry_in}
Remember: a={a_list}, b={b_list}

# Loop i from 0..k (you MUST write all, do not 'skip' out-of-range):
i=0, j={k}-0 -> {in_range0 ? "a[0]="+a0+", b["+j0+"]="+bj0+" -> "+a0+"*"+bj0+" = "+p0 : "b["+j0+"] doesn’t exist → skip"}
i=1, j={k}-1 -> {line for i=1}
...
i={k}, j=0 -> {line for i=k}

S = {carry_in} + {sum_of_in_range_products_or_0s} = {S_total} → output {digit_k}, carry {carry_out}
so far: outputs = [{msb_first_outputs}]
```

Example fill rules


For each i, compute j = k - i.
If 0 ≤ i < n and 0 ≤ j < m, write a[i]=… , b[j]=… -> …*… = ….
Otherwise write “doesn’t exist → skip”.
{sum_of_in_range_products_or_0s} is the textual sum you actually added (omit skipped terms).
{msb_first_outputs} is the running list of output digits with the newest at the front.

# Worked example: 1234 × 567

```
Digits (LSB first): a=[4,3,2,1], b=[7,6,5].

k=0: 
S = carry = 0
Remember: a=[4,3,2,1], b=[7,6,5]
i=0, j=0
a[0]=4, b[0]=7 -> 4*7 = 28
S=0 + 28 = 28 → output 8, carry 2
so far: outputs = [8]

k=1: 
S = carry = 2
Remember: a=[4,3,2,1], b=[7,6,5]
i=0, j=1 -> a[0]=4, b[1]=6 -> 4*6 = 24
i=1, j=0 -> a[1]=3, b[0]=7 -> 3*7 = 21
S= 2 + 24 + 21 = 47 → output 7, carry 4
so far: outputs= [7, 8]

k=2:
S = carry = 4
Remember: a=[4,3,2,1], b=[7,6,5]
i=0, j=2 → a[0]=4, b[2]=5 → 4*5 = 20
i=1, j=1 → a[1]=3, b[1]=6 → 3*6 = 18
i=2, j=0 → a[2]=2, b[0]=7 → 2*7 = 14
S = 4 + 20 + 18 + 14 = 56 → output 6, carry 5
so far: outputs= [6, 7, 8]



k=3:
S = carry = 5
Remember: a=[4,3,2,1], b=[7,6,5]
i=0, j=3 → b[3] doesn’t exist → skip
i=1, j=2 → a[1]=3, b[2]=5 → 15
i=2, j=1 → a[2]=2, b[1]=6 → 12
i=3, j=0 → a[3]=1, b[0]=7 → 7
S = 5 + 15 + 12 + 7 = 39 → output 9, carry 3
so far: outputs= [9, 6, 7, 8]



k=4:
S = carry = 3
Remember: a=[4,3,2,1], b=[7,6,5]
i=0,j=4 -> a[0]=4, b[4] doesn’t exist → skip
i=1,j=3 -> a[1]=3, b[3] doesn’t exist → skip
i=2, j=2 → a[2]=2, b[2]=5 → 10
i=3, j=1 → a[3]=1, b[1]=6 → 6
i=4,j=0 -> a[4] doesn’t exist → skip
S = 3 + 10 + 6 = 19 → output 9, carry 1
so far: outputs= [9, 9, 6, 7, 8]



k=5:
S = carry = 1
Remember: a=[4,3,2,1], b=[7,6,5]
i=0,j=5 -> a[0]=4, b[5] doesn’t exist → skip
i=1,j=4 -> a[1]=3, b[4] doesn’t exist → skip
i=2,j=3 -> a[2]=3, b[3] doesn’t exist → skip
i=3, j=2 → a[3]=1, b[2]=5 → 5
i=4,j=1 -> a[4] doesn't exist -> skip
i=5, j=0 -> a[5] doesn't exist -> skip
S = 1 + 5 = 6 → output 6, carry 0
so far: outputs= [6, 9, 9, 6, 7, 8]

Flush carry: carry=0, done.
Output digits: 699678.

RESULT: 699678
```

# Notes & extensions


- Signs: record the sign; multiply absolute values; apply sign at the end.
- If division/mod by 10 are missing: use a loop: set q=0; while S≥10: S=S-10; q=q+1. Then digit=S, carry=q.
- Be explicit, follow the instructions and use the template shown in the example.
- Return the result at the end in a line starting with "RESULT: "
"""

system_prompt_CoT = "Multiply the two given integers and provide the result.\n Think step by step. Show your work clearly and provide the final result on a line starting with 'RESULT: '."
system_prompt_simple = "Multiply the two given integers and provide the result."

def process_multiplication_task(client: SimpleOpenAIClient, task: MultiplicationTask) -> TaskResult:
    """Process a single multiplication task"""
    try:
        # Choose system prompt based on mode
        if task.mode == "cot":
            system_prompt = system_prompt_CoT
        elif task.mode == "simple":
            system_prompt = system_prompt_simple
        elif task.mode == "with_prompt":
            system_prompt = system_prompt_with_algorithm
        else:
            raise ValueError(f"Unknown mode: {task.mode}")
                
        # Create messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": f"What is {task.x} × {task.y}?"
            }
        ]
        
        # Make API call
        response_text = client.chat_completion(
            messages=messages,
            model="gpt-4o",
            temperature=0,  # Use deterministic responses
            max_tokens=10000,
        )
        
        with open(f"results_7x7_4o/response_{task.mode}_{task.task_id}.txt", "w") as f:
            f.write(response_text + "\n")
        
        # Extract result using robust extraction
        extracted_result = extract_multiplication_result(response_text)
        if extracted_result is None:
            extracted_result = 0
            print(f"Failed to extract result for task {task.task_id} ({task.mode})")
            
        expected_result = task.x * task.y
        is_correct = extracted_result == expected_result
        
        print(f"Task {task.task_id} ({task.mode}): {task.x} × {task.y} = {extracted_result} "
              f"(expected: {expected_result}) {'✓' if is_correct else '✗'}")
        
        return TaskResult(
            task=task,
            response_text=response_text,
            extracted_result=extracted_result,
            expected_result=expected_result,
            is_correct=is_correct
        )
        
    except Exception as e:
        print(f"Error in task {task.task_id}: {str(e)}")
        return TaskResult(
            task=task,
            response_text="",
            extracted_result=0,
            expected_result=task.x * task.y,
            is_correct=False,
            error=str(e)
        )


def run_simple_parallel_experiment(
    num_pairs: int = 5,
    max_workers: int = 2,
    delay_between_requests: float = 1.0
):
    """Run a simple parallel multiplication experiment"""
    
    # Initialize rate limiter and client
    rate_limiter = SimpleRateLimiter(delay_between_requests)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = SimpleOpenAIClient(api_key, rate_limiter)
    
    # Generate tasks (smaller numbers for testing)
    tasks = []
    for i in range(num_pairs):
        x = random.randint(1000000, 9999999)  # Smaller numbers
        y = random.randint(1000000, 9999999)
        
        # Create both with_prompt and cot tasks
        tasks.append(MultiplicationTask(x, y, "with_prompt", i * 3))
        tasks.append(MultiplicationTask(x, y, "cot", i * 3 + 1))
        tasks.append(MultiplicationTask(x, y, "simple", i * 3 + 2))
    
    print(f"Starting experiment with {len(tasks)} tasks, max {max_workers} workers")
    print(f"Delay between requests: {delay_between_requests} seconds")
    print(f"Using robust result extraction")
    
    # Process tasks with thread pool
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_multiplication_task, client, task): task 
            for task in tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            try:
                result = future.result(timeout=60)  # 60 second timeout per task
                results.append(result)

            except Exception as e:
                task = future_to_task[future]
                print(f"Task {task.task_id} failed: {e}")
                results.append(TaskResult(
                    task=task,
                    response_text="",
                    extracted_result=0,
                    expected_result=task.x * task.y,
                    is_correct=False,
                    error=str(e)
                ))
    
    end_time = time.time()
    
    # Analyze results
    corrects_with_prompt = 0
    corrects_cot_prompt = 0
    corrects_simple_prompt = 0
    incorrects_with_prompt = 0
    incorrects_cot_prompt = 0
    incorrects_simple_prompt = 0
    errors = 0
    
    for result in results:
        if result.error:
            errors += 1
            continue
            
        if result.task.mode == "with_prompt":
            if result.is_correct:
                corrects_with_prompt += 1
            else:
                incorrects_with_prompt += 1
        elif result.task.mode == "cot":
            if result.is_correct:
                corrects_cot_prompt += 1
            else:
                incorrects_cot_prompt += 1
        elif result.task.mode == "simple":
            if result.is_correct:
                corrects_simple_prompt += 1
            else:
                incorrects_simple_prompt += 1
        else:  # Unknown mode
            raise ValueError(f"Unknown mode: {result.task.mode}")
    
    # Calculate and display summary
    total_time = end_time - start_time
    total_with_prompt = corrects_with_prompt + incorrects_with_prompt
    total_cot_prompt = corrects_cot_prompt + incorrects_cot_prompt
    total_simple_prompt = corrects_simple_prompt + incorrects_simple_prompt
    
    accuracy_with_prompt = corrects_with_prompt / total_with_prompt if total_with_prompt > 0 else 0
    accuracy_cot_prompt = corrects_cot_prompt / total_cot_prompt if total_cot_prompt > 0 else 0
    accuracy_simple_prompt = corrects_simple_prompt / total_simple_prompt if total_simple_prompt > 0 else 0
    
    summary = f"""
Simple Parallel Multiplication Experiment Results
===============================================
Total tasks: {len(tasks)}
Total time: {total_time:.2f} seconds
Average time per task: {total_time / len(tasks):.2f} seconds
Errors: {errors}

With Algorithm Prompt:
- Correct: {corrects_with_prompt}
- Incorrect: {incorrects_with_prompt}
- Accuracy: {accuracy_with_prompt:.2%}

COT Prompt:
- Correct: {corrects_cot_prompt}
- Incorrect: {incorrects_cot_prompt}
- Accuracy: {accuracy_cot_prompt:.2%}

Simple Prompt:
- Correct: {corrects_simple_prompt}
- Incorrect: {incorrects_simple_prompt}
- Accuracy: {accuracy_simple_prompt:.2%}

Settings:
- Max workers: {max_workers}
- Delay between requests: {delay_between_requests} seconds
"""
    
    print(summary)
    
    with open("results_7x7_4o/simple_parallel_results.txt", "w") as f:
        f.write(summary)
    
    return results


if __name__ == "__main__":
    # Run the experiment
    run_simple_parallel_experiment(
        num_pairs=10,  # Small test
        max_workers=8,  # Conservative
        delay_between_requests=0.5  # 2 second delay between requests
    )
