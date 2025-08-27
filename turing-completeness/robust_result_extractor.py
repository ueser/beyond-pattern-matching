import re
from typing import Optional, List


def extract_multiplication_result(response_text: str) -> Optional[int]:
    """
    Robust extraction of multiplication result from various response formats.
    
    Handles cases like:
    - RESULT: 123456
    - **RESULT: 123456**
    - Result: 123,456,789
    - The answer is 123456
    - 5,074,665,361 × 5,781,107,540 = 29,324,927,927,927,703,940
    - And many other formats
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
    
    # Strategy 6: Last resort - look for any number at the end of the text
    end_number_match = re.search(r'(\d+)\.?\s*$', response_text.strip())
    if end_number_match:
        try:
            return int(end_number_match.group(1))
        except ValueError:
            pass
    
    return None


def test_extraction():
    """Test the extraction function with various formats"""
    
    test_cases = [
        ("RESULT: 123456", 123456),
        ("**RESULT: 13694738584992581688**", 13694738584992581688),
        ("Result: 1,234,567,890", 1234567890),
        ("The answer is 999888777", 999888777),
        ("5,074,665,361 × 5,781,107,540 = 29,324,927,927,927,703,940.", 29324927927927703940),
        ("Let me calculate: 123 × 456 = 56088", 56088),
        ("The product of these numbers is: 987,654,321", 987654321),
        ("Answer: 42", 42),
        ("Some text with numbers 123 and 456 but the result is 56088.", 56088),
        ("Multiple large numbers: 1234567890 and 9876543210 but answer is 12193263111263526900", 12193263111263526900),
    ]
    
    print("Testing result extraction:")
    print("=" * 50)
    
    for i, (text, expected) in enumerate(test_cases, 1):
        result = extract_multiplication_result(text)
        status = "✓" if result == expected else "✗"
        print(f"Test {i}: {status}")
        print(f"  Input: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  Expected: {expected}")
        print(f"  Got: {result}")
        print()


def test_on_real_files():
    """Test extraction on real response files"""
    import os

    results_dir = "results_10x10"
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found")
        return

    print("Testing on real response files:")
    print("=" * 50)

    # Test a few files
    test_files = [
        "response_cot_1.txt",
        "response_simple_2.txt",
        "response_with_prompt_0.txt"
    ]

    for filename in test_files:
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()

            result = extract_multiplication_result(content)
            print(f"File: {filename}")
            print(f"Extracted result: {result}")
            print(f"Content preview: {content[:100]}...")
            print()


if __name__ == "__main__":
    test_extraction()
    print("\n" + "="*50)
    test_on_real_files()
