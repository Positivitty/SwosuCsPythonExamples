#!/usr/bin/env python3
"""
Tower of Hanoi - Complete Solution

This program solves the classic Tower of Hanoi puzzle for n disks,
listing all moves required to transfer all disks from source to destination.

Mathematical Properties:
- Minimum moves required: 2^n - 1
- Recursive structure: T(n) = 2T(n-1) + 1
- Base case: T(1) = 1

Author: Classic recursive solution
"""

from typing import List, Tuple
import time


class TowerOfHanoi:
    """
    Solves the Tower of Hanoi puzzle and tracks all moves.
    """
    
    def __init__(self, n: int, source: str = 'A', destination: str = 'C', auxiliary: str = 'B'):
        """
        Initialize Tower of Hanoi puzzle.
        
        Args:
            n: Number of disks
            source: Name of source peg
            destination: Name of destination peg
            auxiliary: Name of auxiliary peg
        """
        if n < 1:
            raise ValueError("Number of disks must be at least 1")
        
        self.n = n
        self.source = source
        self.destination = destination
        self.auxiliary = auxiliary
        self.moves = []
        self.move_count = 0
        
        # Initialize pegs (stacks) - largest disk at bottom
        self.pegs = {
            source: list(range(n, 0, -1)),  # [n, n-1, ..., 2, 1]
            auxiliary: [],
            destination: []
        }
    
    def solve(self) -> List[Tuple[int, str, str]]:
        """
        Solve the Tower of Hanoi puzzle.
        
        Returns:
            List of moves as tuples: (disk_number, from_peg, to_peg)
        """
        self.moves = []
        self.move_count = 0
        self._hanoi(self.n, self.source, self.destination, self.auxiliary)
        return self.moves
    
    def _hanoi(self, n: int, source: str, destination: str, auxiliary: str):
        """
        Recursive helper function to solve Tower of Hanoi.
        
        Args:
            n: Number of disks to move
            source: Current source peg
            destination: Target destination peg
            auxiliary: Helper auxiliary peg
        """
        if n == 1:
            # Base case: move single disk directly
            self._record_move(1, source, destination)
            return
        
        # Step 1: Move (n-1) disks from source to auxiliary using destination
        self._hanoi(n-1, source, auxiliary, destination)
        
        # Step 2: Move the largest disk from source to destination
        self._record_move(n, source, destination)
        
        # Step 3: Move (n-1) disks from auxiliary to destination using source
        self._hanoi(n-1, auxiliary, destination, source)
    
    def _record_move(self, disk: int, from_peg: str, to_peg: str):
        """
        Record a move and update the peg states.
        
        Args:
            disk: Disk number being moved
            from_peg: Source peg
            to_peg: Destination peg
        """
        self.move_count += 1
        self.moves.append((disk, from_peg, to_peg))
    
    def solve_with_visualization(self) -> List[Tuple[int, str, str]]:
        """
        Solve and track actual peg states for visualization.
        
        Returns:
            List of moves with state tracking
        """
        # Reset pegs
        self.pegs = {
            self.source: list(range(self.n, 0, -1)),
            self.auxiliary: [],
            self.destination: []
        }
        
        self.moves = []
        self.move_count = 0
        self._hanoi_with_state(self.n, self.source, self.destination, self.auxiliary)
        return self.moves
    
    def _hanoi_with_state(self, n: int, source: str, destination: str, auxiliary: str):
        """
        Recursive solver that updates actual peg states.
        """
        if n == 1:
            self._execute_move(source, destination)
            return
        
        self._hanoi_with_state(n-1, source, auxiliary, destination)
        self._execute_move(source, destination)
        self._hanoi_with_state(n-1, auxiliary, destination, source)
    
    def _execute_move(self, from_peg: str, to_peg: str):
        """
        Execute a move by updating peg states.
        """
        if not self.pegs[from_peg]:
            raise RuntimeError(f"Invalid move: {from_peg} is empty")
        
        disk = self.pegs[from_peg].pop()
        
        if self.pegs[to_peg] and self.pegs[to_peg][-1] < disk:
            raise RuntimeError(f"Invalid move: Cannot place disk {disk} on smaller disk {self.pegs[to_peg][-1]}")
        
        self.pegs[to_peg].append(disk)
        self.move_count += 1
        self.moves.append((disk, from_peg, to_peg))
    
    def get_state_string(self) -> str:
        """
        Get current state of all pegs as a string.
        """
        return f"{self.source}={self.pegs[self.source]} {self.auxiliary}={self.pegs[self.auxiliary]} {self.destination}={self.pegs[self.destination]}"
    
    @staticmethod
    def calculate_min_moves(n: int) -> int:
        """
        Calculate minimum number of moves required.
        
        Formula: 2^n - 1
        
        Args:
            n: Number of disks
            
        Returns:
            Minimum number of moves
        """
        return (2 ** n) - 1
    
    @staticmethod
    def verify_solution(n: int, moves: List[Tuple[int, str, str]]) -> bool:
        """
        Verify that a solution is valid.
        
        Args:
            n: Number of disks
            moves: List of moves to verify
            
        Returns:
            True if solution is valid
        """
        # Check move count
        if len(moves) != TowerOfHanoi.calculate_min_moves(n):
            return False
        
        # Simulate the moves
        pegs = {'A': list(range(n, 0, -1)), 'B': [], 'C': []}
        
        for disk, from_peg, to_peg in moves:
            # Check if from_peg has the disk on top
            if not pegs[from_peg] or pegs[from_peg][-1] != disk:
                return False
            
            # Check if move is valid (larger on smaller)
            if pegs[to_peg] and pegs[to_peg][-1] < disk:
                return False
            
            # Execute move
            pegs[from_peg].pop()
            pegs[to_peg].append(disk)
        
        # Check final state
        return pegs['C'] == list(range(n, 0, -1)) and not pegs['A'] and not pegs['B']


def print_moves(n: int, moves: List[Tuple[int, str, str]]):
    """
    Print all moves in a formatted way.
    """
    print(f"\n{'='*60}")
    print(f"Tower of Hanoi Solution for n = {n} disks")
    print(f"{'='*60}")
    print(f"Total moves required: {len(moves)} (Formula: 2^{n} - 1 = {2**n - 1})")
    print(f"{'='*60}\n")
    
    for i, (disk, from_peg, to_peg) in enumerate(moves, 1):
        print(f"Move {i:3d}: Move disk {disk} from {from_peg} to {to_peg}")


def print_moves_with_state(n: int, solver: TowerOfHanoi):
    """
    Print moves with visual representation of peg states.
    """
    print(f"\n{'='*70}")
    print(f"Tower of Hanoi Solution with State Visualization (n = {n})")
    print(f"{'='*70}")
    
    # Initial state
    print(f"\nInitial state: {solver.get_state_string()}")
    print(f"\n{'-'*70}")
    
    moves = solver.solve_with_visualization()
    
    for i, (disk, from_peg, to_peg) in enumerate(moves, 1):
        print(f"Move {i:3d}: Disk {disk} from {from_peg} to {to_peg}")
        print(f"         State: {solver.get_state_string()}")
        if i < len(moves):
            print(f"{'-'*70}")
    
    print(f"\nFinal state: {solver.get_state_string()}")
    print(f"{'='*70}")


def demonstrate_pattern(max_n: int = 5):
    """
    Demonstrate the exponential growth pattern.
    """
    print(f"\n{'='*60}")
    print("Move Count Pattern for Different Values of n")
    print(f"{'='*60}")
    print(f"{'n (disks)':<12} {'Moves Required':<20} {'Formula Check':<20}")
    print(f"{'-'*60}")
    
    for n in range(1, max_n + 1):
        moves_required = TowerOfHanoi.calculate_min_moves(n)
        formula_result = 2**n - 1
        match = "✓" if moves_required == formula_result else "✗"
        print(f"{n:<12} {moves_required:<20} 2^{n} - 1 = {formula_result} {match}")
    
    print(f"{'='*60}\n")


def main():
    """
    Main function to demonstrate Tower of Hanoi solutions.
    """
    print("\n" + "="*70)
    print(" "*20 + "TOWER OF HANOI SOLVER")
    print("="*70)
    
    # Demonstrate pattern
    demonstrate_pattern(7)
    
    # Solve for small values with full output
    for n in [1, 2, 3, 4]:
        solver = TowerOfHanoi(n)
        
        if n <= 3:
            # Show with state visualization for n ≤ 3
            print_moves_with_state(n, solver)
        else:
            # Just show moves for n = 4
            moves = solver.solve()
            print_moves(n, moves)
            
            # Verify solution
            is_valid = TowerOfHanoi.verify_solution(n, moves)
            print(f"\nSolution verification: {'✓ VALID' if is_valid else '✗ INVALID'}")
        
        print()
    
    # Show larger examples (just counts)
    print(f"\n{'='*70}")
    print("For Larger Values:")
    print(f"{'='*70}")
    for n in [5, 10, 15, 20, 30, 64]:
        moves_required = TowerOfHanoi.calculate_min_moves(n)
        print(f"n = {n:2d}: {moves_required:,} moves required")
    
    print(f"\n{'='*70}")
    print("Mathematical Properties:")
    print(f"{'='*70}")
    print("• Recurrence relation: T(n) = 2T(n-1) + 1")
    print("• Closed form: T(n) = 2^n - 1")
    print("• Base case: T(1) = 1")
    print("• Growth rate: Exponential (O(2^n))")
    print("• The solution is unique and optimal")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()