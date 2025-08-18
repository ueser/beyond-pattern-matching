#!/usr/bin/env python3
"""
Brainfuck Step-by-Step Debugger

This script shows the step-by-step execution of a Brainfuck program,
displaying the state of the memory tape, input stream, and output at each step.
"""

from brainfuck import BrainfuckInterpreter

class BrainfuckDebugger(BrainfuckInterpreter):
    """Extended Brainfuck interpreter with step-by-step debugging."""
    
    def __init__(self, memory_size=30, show_memory_range=10):
        super().__init__(memory_size)
        self.show_memory_range = show_memory_range
        self.step_count = 0
        
    def debug_run(self, code, input_data=""):
        """Execute Brainfuck code with step-by-step debugging."""
        print(f"🐛 BRAINFUCK DEBUGGER")
        print(f"Program: {code}")
        print(f"Input: {repr(input_data)} (as chars: {[ord(c) for c in input_data]})")
        print("=" * 80)
        
        # Remove comments (keep only valid BF commands)
        code = ''.join(c for c in code if c in '><+-.,[]!')
        
        input_index = 0
        self.instruction_pointer = 0
        self.output = []
        self.step_count = 0

        # Reset state
        self.memory = [0] * len(self.memory)
        self.pointer = 0

        # Build jump table for brackets
        jump_table = self._build_jump_table(code)
        
        # Show initial state
        self._show_state(code, input_data, input_index, "INITIAL")

        max_steps = 100  # Prevent infinite loops in debugging
        
        while self.instruction_pointer < len(code) and self.step_count < max_steps:
            cmd = code[self.instruction_pointer]
            self.step_count += 1
            
            print(f"\nStep {self.step_count}: Execute '{cmd}' at position {self.instruction_pointer}")
            
            if cmd == '>':
                self.pointer += 1
                if self.pointer >= len(self.memory):
                    self.pointer = 0  # Wrap around
                print(f"  Move pointer right → position {self.pointer}")
                    
            elif cmd == '<':
                self.pointer -= 1
                if self.pointer < 0:
                    self.pointer = len(self.memory) - 1  # Wrap around
                print(f"  Move pointer left → position {self.pointer}")

            elif cmd == '+':
                self.memory[self.pointer] = (self.memory[self.pointer] + 1) % 256
                print(f"  Increment cell[{self.pointer}] → {self.memory[self.pointer]}")
                
            elif cmd == '-':
                self.memory[self.pointer] = (self.memory[self.pointer] - 1) % 256
                print(f"  Decrement cell[{self.pointer}] → {self.memory[self.pointer]}")
                
            elif cmd == '.':
                char = chr(self.memory[self.pointer])
                self.output.append(char)
                print(f"  Output cell[{self.pointer}] = {self.memory[self.pointer]} → '{char}' (ASCII {self.memory[self.pointer]})")
                
            elif cmd == ',':
                if input_index < len(input_data):
                    self.memory[self.pointer] = ord(input_data[input_index])
                    print(f"  Read input[{input_index}] = '{input_data[input_index]}' (ASCII {ord(input_data[input_index])}) → cell[{self.pointer}]")
                    input_index += 1
                else:
                    print(f"  Read input: EOF, cell[{self.pointer}] unchanged")
                    
            elif cmd == '[':
                if self.memory[self.pointer] == 0:
                    new_ip = jump_table[self.instruction_pointer]
                    print(f"  Loop start: cell[{self.pointer}] = 0, jump to position {new_ip}")
                    self.instruction_pointer = new_ip
                else:
                    print(f"  Loop start: cell[{self.pointer}] ≠ 0, enter loop")
                    
            elif cmd == ']':
                if self.memory[self.pointer] != 0:
                    new_ip = jump_table[self.instruction_pointer]
                    print(f"  Loop end: cell[{self.pointer}] ≠ 0, jump back to position {new_ip}")
                    self.instruction_pointer = new_ip
                else:
                    print(f"  Loop end: cell[{self.pointer}] = 0, exit loop")

            elif cmd == '!':
                # Custom extension: copy cell to input
                cell_value = self.memory[self.pointer]
                if 0 <= cell_value <= 255:
                    input_data += chr(cell_value)
                    print(f"  Copy cell[{self.pointer}] = {cell_value} → input stream")

            self.instruction_pointer += 1
            
            # Show state after each step
            self._show_state(code, input_data, input_index, f"AFTER STEP {self.step_count}")

        if self.step_count >= max_steps:
            print(f"\n⚠️ Execution stopped after {max_steps} steps (possible infinite loop)")
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"Output: {''.join(self.output)} → {[ord(c) for c in self.output]}")
        return ''.join(self.output)

    def _show_state(self, code, input_data, input_index, label):
        """Show current state of memory, pointer, and program."""
        print(f"\n{label}:")
        
        # Show program with instruction pointer
        program_display = ""
        for i, cmd in enumerate(code):
            if i == self.instruction_pointer:
                program_display += f"[{cmd}]"
            else:
                program_display += cmd
        print(f"Program:  {program_display}")
        
        # Show input stream
        input_display = ""
        for i, char in enumerate(input_data):
            if i == input_index:
                input_display += f"[{char}]"
            else:
                input_display += char
        if input_index >= len(input_data):
            input_display += "[EOF]"
        print(f"Input:    {input_display}")
        
        # Show memory tape (focused around pointer)
        start = max(0, self.pointer - self.show_memory_range // 2)
        end = min(len(self.memory), start + self.show_memory_range)
        
        # Adjust start if we're near the end
        if end - start < self.show_memory_range:
            start = max(0, end - self.show_memory_range)
            
        memory_vals = []
        memory_ptrs = []
        memory_addrs = []
        
        for i in range(start, end):
            memory_vals.append(f"{self.memory[i]:3d}")
            memory_ptrs.append(" ^ " if i == self.pointer else "   ")
            memory_addrs.append(f"{i:3d}")
            
        print(f"Memory:   [" + "|".join(memory_vals) + "]")
        print(f"Pointer:   " + " ".join(memory_ptrs))
        print(f"Address:   " + " ".join(memory_addrs))
        
        # Show output so far
        if self.output:
            output_chars = ''.join(self.output)
            output_nums = [ord(c) for c in self.output]
            print(f"Output:   '{output_chars}' → {output_nums}")
        else:
            print(f"Output:   (empty)")


def main():
    """Interactive debugger."""
    debugger = BrainfuckDebugger(memory_size=20, show_memory_range=8)
    
    print("🧠 Brainfuck Step-by-Step Debugger")
    print("Enter 'quit' to exit\n")
    
    while True:
        print("-" * 60)
        program = input("Enter Brainfuck program: ").strip()
        if program.lower() == 'quit':
            break
            
        input_data = input("Enter input data: ").strip()
        
        try:
            print()
            result = debugger.debug_run(program, input_data)
            print(f"\n✅ Execution complete. Final output: '{result}'")
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")


if __name__ == "__main__":
    # Example runs if called directly
    debugger = BrainfuckDebugger(memory_size=15, show_memory_range=6)
    
    print("🧪 EXAMPLE 1: Simple increment (f(x) = x + 1)")
    debugger.debug_run(",+.", chr(3))
    
    print("\n" + "="*80)
    print("\n🧪 EXAMPLE 2: Doubling program (f(x) = 2*x)")  
    debugger.debug_run(",[>++<-]>.", chr(3))
    
    print("\n" + "="*80)
    print("\n🧪 EXAMPLE 3: Interactive mode")
    main()