import json
import os
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import requests
import re

load_dotenv()


@dataclass
class MultiplicationTask:
    """Represents a single multiplication task"""
    x: int
    y: int
    mode: str  # "with_prompt" or "without_prompt"
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
        
    def chat_completion(self, messages: List[Dict], model: str = "gpt-4o-mini", **kwargs) -> str:
        """Make a chat completion request"""
        self.rate_limiter.acquire()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_p": kwargs.get("top_p", 1),
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")


# System prompts
system_prompt_with_algorithm = """You are a multiplication expert. Multiply the two given integers step by step using the standard multiplication algorithm. Show your work clearly and provide the final result on a line starting with "RESULT: "."""

system_prompt_simple = "Multiply the two given integers and provide the result."


def process_multiplication_task(client: SimpleOpenAIClient, task: MultiplicationTask) -> TaskResult:
    """Process a single multiplication task"""
    try:
        # Choose system prompt based on mode
        system_prompt = system_prompt_with_algorithm if task.mode == "with_prompt" else system_prompt_simple
        
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
            model="gpt-4o-mini",
            temperature=0,  # Use deterministic responses
            max_tokens=4096,
        )
        
        # Extract result using robust extraction
        extracted_result = extract_multiplication_result(response_text)
        if extracted_result is None:
            extracted_result = 0
            
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


def run_improved_parallel_experiment(
    num_pairs: int = 5,
    max_workers: int = 2,
    delay_between_requests: float = 1.0
):
    """Run an improved parallel multiplication experiment"""
    
    # Initialize rate limiter and client
    rate_limiter = SimpleRateLimiter(delay_between_requests)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = SimpleOpenAIClient(api_key, rate_limiter)
    
    # Generate tasks (medium-sized numbers for testing)
    tasks = []
    for i in range(num_pairs):
        x = random.randint(10000, 999999)  # 5-6 digit numbers
        y = random.randint(10000, 999999)
        
        # Create both with_prompt and without_prompt tasks
        tasks.append(MultiplicationTask(x, y, "with_prompt", i * 2))
        tasks.append(MultiplicationTask(x, y, "without_prompt", i * 2 + 1))
    
    print(f"Starting improved experiment with {len(tasks)} tasks, max {max_workers} workers")
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
    corrects_without_prompt = 0
    incorrects_with_prompt = 0
    incorrects_without_prompt = 0
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
        else:
            if result.is_correct:
                corrects_without_prompt += 1
            else:
                incorrects_without_prompt += 1
    
    # Save individual responses
    for result in results:
        if not result.error:
            filename = f"improved_response_{result.task.mode}_{result.task.task_id}.txt"
            with open(filename, "w") as f:
                f.write(result.response_text + "\n")
    
    # Calculate and display summary
    total_time = end_time - start_time
    total_with_prompt = corrects_with_prompt + incorrects_with_prompt
    total_without_prompt = corrects_without_prompt + incorrects_without_prompt
    
    accuracy_with_prompt = corrects_with_prompt / total_with_prompt if total_with_prompt > 0 else 0
    accuracy_without_prompt = corrects_without_prompt / total_without_prompt if total_without_prompt > 0 else 0
    
    summary = f"""
Improved Parallel Multiplication Experiment Results
================================================
Total tasks: {len(tasks)}
Total time: {total_time:.2f} seconds
Average time per task: {total_time / len(tasks):.2f} seconds
Errors: {errors}

With Algorithm Prompt:
- Correct: {corrects_with_prompt}
- Incorrect: {incorrects_with_prompt}
- Accuracy: {accuracy_with_prompt:.2%}

Without Algorithm Prompt:
- Correct: {corrects_without_prompt}
- Incorrect: {incorrects_without_prompt}
- Accuracy: {accuracy_without_prompt:.2%}

Settings:
- Max workers: {max_workers}
- Delay between requests: {delay_between_requests} seconds
- Using robust result extraction
"""
    
    print(summary)
    
    with open("improved_parallel_results.txt", "w") as f:
        f.write(summary)
    
    return results


if __name__ == "__main__":
    # Run the experiment
    run_improved_parallel_experiment(
        num_pairs=3,  # Small test
        max_workers=2,  # Conservative
        delay_between_requests=2.0  # 2 second delay between requests
    )
