#!/usr/bin/env python3
"""
Brainfuck Interpreter

Brainfuck is an esoteric programming language with only 8 commands:
    >   Move the pointer to the right
    <   Move the pointer to the left
    +   Increment the memory cell at the pointer
    -   Decrement the memory cell at the pointer
    .   Output the character signified by the cell at the pointer
    ,   Input a character and store it in the cell at the pointer
    [   Jump past the matching ] if the cell at the pointer is 0
    ]   Jump back to the matching [ if the cell at the pointer is nonzero
    !   Copy the value from the cell at the pointer to the input data stream

All other characters are treated as comments and ignored.
"""

class BrainfuckInterpreter:
    def __init__(self, memory_size=30000):
        self.memory = [0] * memory_size
        self.pointer = 0
        self.instruction_pointer = 0
        self.output = []
        self.input_reads = 0
        self.output_writes = 0
        self.hit_step_limit = False

    def run(self, code, input_data="", debug=False):
        """Execute Brainfuck code with optional input data."""
        # Remove comments (keep only valid BF commands)
        code = ''.join(c for c in code if c in '><+-.,[]!')

        input_index = 0
        self.instruction_pointer = 0
        self.output = []
        self.input_reads = 0
        self.output_writes = 0
        self.hit_step_limit = False

        # Reset state
        self.memory = [0] * len(self.memory)
        self.pointer = 0

        # Build jump table for brackets
        jump_table = self._build_jump_table(code)

        step_count = 0
        max_steps = 10000  # Prevent infinite loops

        while self.instruction_pointer < len(code) and step_count < max_steps:
            cmd = code[self.instruction_pointer]

            if debug and step_count < 50:  # Only show first 50 steps
                print(f"Step {step_count:2d}: IP={self.instruction_pointer:2d} CMD='{cmd}' PTR={self.pointer} CELL={self.memory[self.pointer]} MEM={self.memory[:5]}")

            if cmd == '>':
                self.pointer += 1
                if self.pointer >= len(self.memory):
                    # Circular tape assume wrapping
                    self.pointer = 0
                    
            elif cmd == '<':
                self.pointer -= 1
                if self.pointer < 0:
                    # Circular tape assume wrapping
                    self.pointer = len(self.memory) - 1

            elif cmd == '+':
                self.memory[self.pointer] = (self.memory[self.pointer] + 1) % 256
                
            elif cmd == '-':
                self.memory[self.pointer] = (self.memory[self.pointer] - 1) % 256
                
            elif cmd == '.':
                self.output.append(chr(self.memory[self.pointer]))
                self.output_writes += 1
                
            elif cmd == ',':
                if len(input_data) > 0:
                    # Circular input: cycle back to beginning when reaching end
                    self.memory[self.pointer] = ord(input_data[input_index % len(input_data)])
                    input_index += 1  # Move to next input character
                    self.input_reads += 1
                else:
                    # No input data: leave cell unchanged
                    pass
                
            elif cmd == '[':
                if self.memory[self.pointer] == 0:
                    self.instruction_pointer = jump_table[self.instruction_pointer]
                    
            elif cmd == ']':
                if self.memory[self.pointer] != 0:
                    self.instruction_pointer = jump_table[self.instruction_pointer]

            elif cmd == '!':
                # Append current cell value to input stream (with bounds checking)
                cell_value = self.memory[self.pointer]
                if 0 <= cell_value <= 255:
                    input_data += chr(cell_value)
                else:
                    # Handle out-of-bounds values gracefully
                    input_data += chr(cell_value % 256)

            self.instruction_pointer += 1
            step_count += 1

        if step_count >= max_steps:
            self.hit_step_limit = True

        return ''.join(self.output)

    def _build_jump_table(self, code):
        """Build a table mapping bracket positions for efficient jumping."""
        jump_table = {}
        stack = []

        for i, cmd in enumerate(code):
            if cmd == '[':
                stack.append(i)
            elif cmd == ']':
                if not stack:
                    raise SyntaxError(f"Unmatched ']' at position {i}")
                start = stack.pop()
                jump_table[start] = i
                jump_table[i] = start

        if stack:
            raise SyntaxError(f"Unmatched '[' at position {stack[-1]}")

        return jump_table