import random

TOK = "><+-[],."

def is_balanced(s: str) -> bool:
    """Check if brackets are balanced."""
    d = 0
    for c in s:
        if c == '[': 
            d += 1
        elif c == ']': 
            d -= 1
            if d < 0: 
                return False
    return d == 0

def random_program(max_len=40):
    """Generate random BF program with structure ,<body>."""
    body_len = random.randint(1, max_len-2)
    body, depth = [], 0
    for _ in range(body_len):
        if depth > 0 and random.random() < 0.25:
            body.append(']')
            depth -= 1
        else:
            c = random.choice("><+-[],")
            if c == '[': 
                depth += 1
            body.append(c)
    body += [']'] * depth
    return ',' + ''.join(body) + '.'

def mutate(s: str, p=0.3, max_len=40):
    """Mutate program with point mutations, insertions, deletions."""
    if random.random() > p: 
        return s
    body = list(s[1:-1])
    op = random.random()
    if op < 0.33 and body:
        # Point mutation
        i = random.randrange(len(body))
        body[i] = random.choice("><+-[],")
    elif op < 0.66 and len(body) < max_len - 2:
        # Insertion
        i = random.randrange(len(body) + 1)
        body.insert(i, random.choice("><+-[],"))
    elif body:
        # Deletion
        i = random.randrange(len(body))
        body.pop(i)
    out = ',' + ''.join(body[:max_len-2]) + '.'
    return out if is_balanced(out) else s

def crossover(a: str, b: str, max_len=40):
    """Single-point crossover between two programs."""
    A, B = a[1:-1], b[1:-1]
    i = random.randrange(len(A) + 1)
    j = random.randrange(len(B) + 1)
    child = ',' + (A[:i] + B[j:])[:max_len-2] + '.'
    return child if is_balanced(child) else mutate(child, 1.0, max_len)