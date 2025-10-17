#!/usr/bin/env python3
"""
Birthday Paradox Problem

Determines the minimum number of people needed to achieve various probability
thresholds for at least two people sharing the same birthday.

The Birthday Paradox (or Birthday Problem) is a classic probability puzzle
that demonstrates how our intuition about probability can be misleading.
"""

import random
from typing import List, Tuple
import math


class BirthdayParadox:
    """
    Calculates and verifies the Birthday Paradox probabilities.
    """
    
    def __init__(self, days_in_year: int = 365):
        """
        Initialize with number of days in a year.
        
        Args:
            days_in_year: Number of possible birthdays (default 365, ignoring leap years)
        """
        self.days = days_in_year
    
    def probability_no_match(self, n: int) -> float:
        """
        Calculate probability that NO two people share a birthday.
        
        Formula: P(no match) = (365/365) × (364/365) × (363/365) × ... × ((365-n+1)/365)
        
        Args:
            n: Number of people
            
        Returns:
            Probability that no two people share a birthday
        """
        if n > self.days:
            return 0.0  # Pigeonhole principle: guaranteed match
        
        probability = 1.0
        for i in range(n):
            probability *= (self.days - i) / self.days
        
        return probability
    
    def probability_at_least_one_match(self, n: int) -> float:
        """
        Calculate probability that AT LEAST two people share a birthday.
        
        P(at least one match) = 1 - P(no match)
        
        Args:
            n: Number of people
            
        Returns:
            Probability that at least two people share a birthday
        """
        return 1.0 - self.probability_no_match(n)
    
    def find_minimum_people(self, target_probability: float) -> int:
        """
        Find minimum number of people needed to achieve target probability.
        
        Args:
            target_probability: Desired probability threshold (e.g., 0.70 for 70%)
            
        Returns:
            Minimum number of people needed
        """
        n = 1
        while self.probability_at_least_one_match(n) < target_probability:
            n += 1
        return n
    
    def monte_carlo_simulation(self, n: int, num_trials: int = 100000) -> float:
        """
        Verify probability through Monte Carlo simulation.
        
        Args:
            n: Number of people in each trial
            num_trials: Number of simulation trials to run
            
        Returns:
            Empirical probability from simulation
        """
        matches = 0
        
        for _ in range(num_trials):
            # Generate random birthdays for n people
            birthdays = [random.randint(1, self.days) for _ in range(n)]
            
            # Check if any two people share a birthday
            if len(birthdays) != len(set(birthdays)):
                matches += 1
        
        return matches / num_trials
    
    def analyze_thresholds(self, thresholds: List[float]) -> List[Tuple[float, int, float, float]]:
        """
        Analyze multiple probability thresholds.
        
        Args:
            thresholds: List of target probabilities
            
        Returns:
            List of tuples: (threshold, min_people, theoretical_prob, simulated_prob)
        """
        results = []
        
        for threshold in thresholds:
            n = self.find_minimum_people(threshold)
            theoretical = self.probability_at_least_one_match(n)
            simulated = self.monte_carlo_simulation(n, num_trials=50000)
            results.append((threshold, n, theoretical, simulated))
        
        return results


def main():
    """
    Main function to solve the Birthday Paradox problem.
    """
    print("=" * 70)
    print("BIRTHDAY PARADOX PROBLEM")
    print("=" * 70)
    print("\nProblem: Find the minimum number of people needed so that the")
    print("probability at least two share a birthday meets various thresholds.\n")
    
    # Initialize the calculator
    calculator = BirthdayParadox(days_in_year=365)
    
    # Target probability thresholds
    thresholds = [0.70, 0.80, 0.90, 0.95, 0.99]
    
    print("THEORETICAL ANALYSIS:")
    print("-" * 70)
    print(f"{'Target Prob':<15} {'Min People':<12} {'Actual Prob':<15} {'Percentage':<12}")
    print("-" * 70)
    
    theoretical_results = []
    for threshold in thresholds:
        n = calculator.find_minimum_people(threshold)
        actual_prob = calculator.probability_at_least_one_match(n)
        theoretical_results.append((threshold, n, actual_prob))
        
        print(f"{threshold*100:>6.0f}%          {n:<12} {actual_prob:<15.6f} {actual_prob*100:>6.2f}%")
    
    # Show the formula and explanation
    print("\n" + "=" * 70)
    print("MATHEMATICAL EXPLANATION:")
    print("=" * 70)
    print("\nThe probability is calculated using the complement:")
    print("  P(at least one match) = 1 - P(no matches)")
    print("\nP(no matches) = (365/365) × (364/365) × (363/365) × ... × ((365-n+1)/365)")
    print("\nThis is because:")
    print("  - First person: can have any birthday (365/365)")
    print("  - Second person: must avoid 1 birthday (364/365)")
    print("  - Third person: must avoid 2 birthdays (363/365)")
    print("  - And so on...")
    
    # Verify with Monte Carlo simulation
    print("\n" + "=" * 70)
    print("MONTE CARLO VERIFICATION (50,000 trials per case):")
    print("=" * 70)
    print(f"{'Target':<10} {'People':<10} {'Theoretical':<15} {'Simulated':<15} {'Difference':<12}")
    print("-" * 70)
    
    for threshold, n, theoretical in theoretical_results:
        simulated = calculator.monte_carlo_simulation(n, num_trials=50000)
        diff = abs(theoretical - simulated)
        
        print(f"{threshold*100:>5.0f}%     {n:<10} {theoretical*100:>6.2f}%          "
              f"{simulated*100:>6.2f}%          {diff*100:>5.2f}%")
    
    # Additional interesting facts
    print("\n" + "=" * 70)
    print("INTERESTING FACTS:")
    print("=" * 70)
    print(f"\n• With just 23 people, there's a {calculator.probability_at_least_one_match(23)*100:.1f}% chance!")
    print(f"• With 50 people, the probability reaches {calculator.probability_at_least_one_match(50)*100:.1f}%")
    print(f"• With 70 people, the probability is {calculator.probability_at_least_one_match(70)*100:.2f}%")
    print(f"• You need {calculator.find_minimum_people(1.0)} people to guarantee a match (Pigeonhole Principle)")
    
    # Detailed breakdown for one case
    print("\n" + "=" * 70)
    print("DETAILED CALCULATION EXAMPLE (50% threshold):")
    print("=" * 70)
    n_50 = calculator.find_minimum_people(0.50)
    prob_no_match = calculator.probability_no_match(n_50)
    prob_match = calculator.probability_at_least_one_match(n_50)
    
    print(f"\nFor n = {n_50} people:")
    print(f"  P(no match) = {prob_no_match:.6f} = {prob_no_match*100:.2f}%")
    print(f"  P(at least one match) = 1 - {prob_no_match:.6f}")
    print(f"                        = {prob_match:.6f} = {prob_match*100:.2f}%")
    
    # Try n-1 to show it's the minimum
    prob_match_prev = calculator.probability_at_least_one_match(n_50 - 1)
    print(f"\nWith {n_50-1} people: probability = {prob_match_prev*100:.2f}% (below 50%)")
    print(f"With {n_50} people: probability = {prob_match*100:.2f}% (above 50%)")
    print(f"Therefore, {n_50} is the minimum number needed.")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("\nThe Birthday Paradox demonstrates that our intuition about")
    print("probability can be surprisingly inaccurate. The required number")
    print("of people is much smaller than most people expect!")
    print("\nThe Monte Carlo simulations closely match the theoretical values,")
    print("confirming our mathematical analysis is correct.")
    print("=" * 70)


if __name__ == "__main__":
    main()