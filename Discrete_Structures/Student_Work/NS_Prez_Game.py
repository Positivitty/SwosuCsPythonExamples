#!/usr/bin/env python3
"""
Presidents (Asshole) Card Game
==============================

A fully playable command-line implementation of the Presidents card game with AI players,
configurable rules, and multi-round tournaments.

Rules:
- Standard 52-card deck, suits irrelevant for comparisons
- Rank order: 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2
- Legal plays: singles, pairs, triples, four-of-a-kind
- Must match combo size and play equal or higher rank
- Round 1 starts with 3♣, subsequent rounds with President leading
- Roles assigned by finish order: President, VP, Neutral, Vice Asshole, Asshole
- Optional inter-round trading between President↔Asshole and VP↔Vice Asshole

Variants (toggleable):
- --burn-on-equal: Equal rank plays clear the pile
- --twos-clear: Playing a 2 immediately clears the pile
- --start-with-lowest: President leads after Round 1 (default: on)
- --target-points N: Game ends when someone reaches N points
- --rounds N: Play exactly N rounds

Usage:
    python presidents.py --players 5 --humans 1 --rounds 7 --burn-on-equal --seed 42
    python presidents.py --self-test

Examples:
    python presidents.py                           # 4 players, 1 human, 5 rounds
    python presidents.py --players 6 --humans 2   # 6 players, 2 humans
    python presidents.py --burn-on-equal --twos-clear  # Enable variants
    python presidents.py --target-points 20       # First to 20 points wins
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Set, Tuple, Union


class Suit(Enum):
    """Card suits (irrelevant for gameplay but needed for deck)."""
    CLUBS = "♣"
    DIAMONDS = "♦"
    HEARTS = "♥"
    SPADES = "♠"


class Rank(IntEnum):
    """Card ranks in ascending order of power."""
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    TWO = 15  # Highest rank

    def __str__(self) -> str:
        if self == Rank.JACK:
            return "J"
        elif self == Rank.QUEEN:
            return "Q"
        elif self == Rank.KING:
            return "K"
        elif self == Rank.ACE:
            return "A"
        elif self == Rank.TWO:
            return "2"
        else:
            return str(self.value)


class Role(IntEnum):
    """Player roles with associated points."""
    ASSHOLE = -1
    VICE_ASSHOLE = 0
    NEUTRAL = 1
    VICE_PRESIDENT = 2
    PRESIDENT = 3


@dataclass(frozen=True)
class Card:
    """Represents a playing card."""
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit.value}"

    def __lt__(self, other: 'Card') -> bool:
        return self.rank < other.rank


@dataclass(frozen=True)
class Combo:
    """Represents a combination of cards played together."""
    cards: Tuple[Card, ...]
    rank: Rank
    size: int

    @classmethod
    def from_cards(cls, cards: List[Card]) -> Optional['Combo']:
        """Create a combo from a list of cards if valid."""
        if not cards:
            return None
        
        ranks = [card.rank for card in cards]
        if len(set(ranks)) != 1:  # All cards must have same rank
            return None
            
        return cls(
            cards=tuple(sorted(cards)),
            rank=ranks[0],
            size=len(cards)
        )

    def beats(self, other: Optional['Combo']) -> bool:
        """Check if this combo beats another combo."""
        if other is None:
            return True
        # Must be same size and higher or equal rank
        return self.size == other.size and self.rank >= other.rank

    def __str__(self) -> str:
        return " ".join(str(card) for card in self.cards)


class Deck:
    """Standard 52-card deck."""
    
    def __init__(self) -> None:
        self.cards: List[Card] = []
        self.reset()
    
    def reset(self) -> None:
        """Reset deck to full 52 cards."""
        self.cards = [
            Card(rank, suit) 
            for rank in Rank 
            for suit in Suit
        ]
    
    def shuffle(self) -> None:
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def deal(self, num_cards: int) -> List[Card]:
        """Deal specified number of cards from top of deck."""
        dealt = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt


class Hand:
    """Player's hand of cards with analysis capabilities."""
    
    def __init__(self, cards: List[Card]) -> None:
        self.cards = sorted(cards)
    
    def remove_cards(self, cards: List[Card]) -> None:
        """Remove specified cards from hand."""
        for card in cards:
            if card in self.cards:
                self.cards.remove(card)
    
    def add_cards(self, cards: List[Card]) -> None:
        """Add cards to hand and keep sorted."""
        self.cards.extend(cards)
        self.cards.sort()
    
    def get_legal_combos(self, current_combo: Optional[Combo]) -> List[Combo]:
        """Get all legal combinations that can beat the current combo."""
        from itertools import combinations
        legal_combos = []
        
        # Group cards by rank
        rank_groups = defaultdict(list)
        for card in self.cards:
            rank_groups[card.rank].append(card)
        
        # If leading, can play any size combo (1-4)
        if current_combo is None:
            for rank, cards_of_rank in rank_groups.items():
                # Generate all possible combo sizes for this rank
                for size in range(1, min(len(cards_of_rank) + 1, 5)):  # Max 4 of a kind
                    for combo_cards in combinations(cards_of_rank, size):
                        combo = Combo.from_cards(list(combo_cards))
                        if combo:
                            legal_combos.append(combo)
        else:
            # Must match the current combo size and beat its rank
            target_size = current_combo.size
            min_rank = current_combo.rank
            
            for rank, cards_of_rank in rank_groups.items():
                if len(cards_of_rank) >= target_size and rank >= min_rank:
                    # Generate ALL combinations of the target size from available cards
                    for combo_cards in combinations(cards_of_rank, target_size):
                        combo = Combo.from_cards(list(combo_cards))
                        if combo and combo.beats(current_combo):
                            legal_combos.append(combo)
        
        return legal_combos
    
    def get_analysis(self) -> Dict[int, List[Rank]]:
        """Analyze hand into singles, pairs, triples, and quads."""
        rank_counts = Counter(card.rank for card in self.cards)
        analysis = defaultdict(list)
        
        for rank, count in rank_counts.items():
            analysis[count].append(rank)
        
        return dict(analysis)
    
    def has_three_of_clubs(self) -> bool:
        """Check if hand contains 3 of clubs."""
        return any(card.rank == Rank.THREE and card.suit == Suit.CLUBS 
                  for card in self.cards)
    
    def get_highest_cards(self, count: int) -> List[Card]:
        """Get the highest N cards from hand."""
        return sorted(self.cards, reverse=True)[:count]
    
    def is_empty(self) -> bool:
        """Check if hand is empty."""
        return len(self.cards) == 0
    
    def __len__(self) -> int:
        return len(self.cards)
    
    def __str__(self) -> str:
        return " ".join(str(card) for card in self.cards)


class Player:
    """Base player class."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.hand = Hand([])
        self.role = Role.NEUTRAL
        self.total_points = 0
        self.finished_position: Optional[int] = None
        self.passed_this_trick = False
    
    def reset_for_round(self) -> None:
        """Reset player state for a new round."""
        self.finished_position = None
        self.passed_this_trick = False
    
    def choose_play(self, current_combo: Optional[Combo], 
                   game_state: Dict) -> Optional[Combo]:
        """Choose a play. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def choose_cards_to_give(self, count: int) -> List[Card]:
        """Choose cards to give in inter-round trading."""
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"{self.name} ({self.role.name})"


class HumanPlayer(Player):
    """Human player with CLI interaction."""
    
    def choose_play(self, current_combo: Optional[Combo], 
                   game_state: Dict) -> Optional[Combo]:
        """Let human choose their play via CLI."""
        legal_combos = self.hand.get_legal_combos(current_combo)
        
        if not legal_combos:
            print(f"\n{self.name}, you have no legal plays and must pass.")
            input("Press Enter to continue...")
            return None
        
        print(f"\n{self.name}, it's your turn!")
        print(f"Your hand: {self.hand}")
        
        analysis = self.hand.get_analysis()
        if analysis:
            print("Hand analysis:")
            for size, ranks in analysis.items():
                size_name = {1: "Singles", 2: "Pairs", 3: "Triples", 4: "Quads"}[size]
                print(f"  {size_name}: {', '.join(str(r) for r in sorted(ranks))}")
        
        if current_combo:
            print(f"Current combo: {current_combo} (size {current_combo.size})")
        else:
            print("You are leading - play any combo")
        
        print("\nLegal plays:")
        for i, combo in enumerate(legal_combos, 1):
            print(f"  {i}. {combo}")
        print(f"  {len(legal_combos) + 1}. Pass")
        
        while True:
            try:
                choice = int(input("Choose your play (number): "))
                if 1 <= choice <= len(legal_combos):
                    return legal_combos[choice - 1]
                elif choice == len(legal_combos) + 1:
                    return None
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Please enter a number.")
    
    def choose_cards_to_give(self, count: int) -> List[Card]:
        """Let human choose cards to give away."""
        print(f"\n{self.name}, choose {count} card{'s' if count > 1 else ''} to give:")
        print(f"Your hand: {self.hand}")
        
        chosen_cards = []
        available_cards = self.hand.cards.copy()
        
        for i in range(count):
            print(f"\nChoose card {i+1} of {count}:")
            for j, card in enumerate(available_cards, 1):
                print(f"  {j}. {card}")
            
            while True:
                try:
                    choice = int(input("Choose card (number): "))
                    if 1 <= choice <= len(available_cards):
                        card = available_cards.pop(choice - 1)
                        chosen_cards.append(card)
                        break
                    else:
                        print("Invalid choice. Try again.")
                except ValueError:
                    print("Please enter a number.")
        
        return chosen_cards


class AIPlayer(Player):
    """AI player with configurable difficulty."""
    
    def __init__(self, name: str, difficulty: str = "medium") -> None:
        super().__init__(name)
        self.difficulty = difficulty
    
    def choose_play(self, current_combo: Optional[Combo], 
                   game_state: Dict) -> Optional[Combo]:
        """Choose AI play based on heuristics."""
        legal_combos = self.hand.get_legal_combos(current_combo)
        
        if not legal_combos:
            return None
        
        if self.difficulty == "easy":
            return self._choose_easy(legal_combos, current_combo, game_state)
        else:
            return self._choose_medium(legal_combos, current_combo, game_state)
    
    def _choose_easy(self, legal_combos: List[Combo], 
                    current_combo: Optional[Combo], 
                    game_state: Dict) -> Combo:
        """Simple AI that plays randomly from legal options."""
        return random.choice(legal_combos)
    
    def _choose_medium(self, legal_combos: List[Combo], 
                      current_combo: Optional[Combo], 
                      game_state: Dict) -> Combo:
        """Smarter AI with basic strategy."""
        burn_on_equal = game_state.get("burn_on_equal", False)
        
        # If leading, prefer to dump low cards
        if current_combo is None:
            return min(legal_combos, key=lambda c: (c.rank, -c.size))
        
        # Look for exact matches to burn if rule is enabled
        if burn_on_equal and current_combo:
            exact_matches = [c for c in legal_combos if c.rank == current_combo.rank]
            if exact_matches:
                return random.choice(exact_matches)
        
        # Save 2s for later unless hand is very small
        if len(self.hand) > 3:
            non_twos = [c for c in legal_combos if c.rank != Rank.TWO]
            if non_twos:
                legal_combos = non_twos
        
        # Save big sets (triples/quads) unless hand is small
        if len(self.hand) > 5:
            small_combos = [c for c in legal_combos if c.size <= 2]
            if small_combos:
                legal_combos = small_combos
        
        # Play the lowest legal combo that beats the pile
        return min(legal_combos, key=lambda c: (c.rank, c.size))
    
    def choose_cards_to_give(self, count: int) -> List[Card]:
        """AI chooses highest cards to give away."""
        return self.hand.get_highest_cards(count)


class Table:
    """Manages the current trick and pile state."""
    
    def __init__(self) -> None:
        self.current_combo: Optional[Combo] = None
        self.last_player: Optional[Player] = None
        self.players_in_trick: Set[Player] = set()
    
    def play_combo(self, player: Player, combo: Combo) -> bool:
        """Play a combo to the table. Returns True if trick clears."""
        self.current_combo = combo
        self.last_player = player
        self.players_in_trick.add(player)
        
        # Four of a kind always clears
        return combo.size == 4
    
    def player_passes(self, player: Player) -> None:
        """Mark player as passing this trick."""
        player.passed_this_trick = True
    
    def clear_trick(self) -> Player:
        """Clear the current trick and return the winner."""
        winner = self.last_player
        self.current_combo = None
        self.last_player = None
        self.players_in_trick.clear()
        return winner
    
    def reset_passes(self, players: List[Player]) -> None:
        """Reset all players' pass status for new trick."""
        for player in players:
            player.passed_this_trick = False


class Game:
    """Main game controller."""
    
    def __init__(self, num_players: int, num_humans: int, **options) -> None:
        self.num_players = num_players
        self.players: List[Player] = []
        self.table = Table()
        self.round_num = 0
        self.options = options
        
        # Create players
        for i in range(num_humans):
            self.players.append(HumanPlayer(f"Human{i+1}"))
        
        ai_difficulty = options.get("ai_difficulty", "medium")
        for i in range(num_players - num_humans):
            self.players.append(AIPlayer(f"AI{i+1}", ai_difficulty))
        
        random.shuffle(self.players)  # Random seating
    
    def play_game(self) -> None:
        """Play the complete game."""
        target_rounds = self.options.get("rounds", 5)
        target_points = self.options.get("target_points")
        
        print(f"\n🎮 Starting Presidents game with {self.num_players} players!")
        print("Players:", ", ".join(p.name for p in self.players))
        
        while True:
            self.round_num += 1
            print(f"\n{'='*50}")
            print(f"ROUND {self.round_num}")
            print(f"{'='*50}")
            
            self.play_round()
            self.show_round_summary()
            
            # Check end conditions
            if target_points:
                max_points = max(p.total_points for p in self.players)
                if max_points >= target_points:
                    break
            elif self.round_num >= target_rounds:
                break
        
        self.show_final_results()
    
    def play_round(self) -> None:
        """Play a single round."""
        # Deal cards
        deck = Deck()
        deck.shuffle()
        
        cards_per_player = 52 // self.num_players
        for player in self.players:
            player.hand = Hand(deck.deal(cards_per_player))
            player.reset_for_round()
        
        # Inter-round trading (except first round)
        if self.round_num > 1 and self.options.get("trading", True):
            self.conduct_trading()
        
        # Determine starting player
        if self.round_num == 1:
            # Find player with 3 of clubs
            current_player_idx = next(
                i for i, p in enumerate(self.players) 
                if p.hand.has_three_of_clubs()
            )
        else:
            # President starts (if still in game)
            president = next((p for p in self.players if p.role == Role.PRESIDENT), None)
            if president:
                current_player_idx = self.players.index(president)
            else:
                current_player_idx = 0
        
        active_players = self.players.copy()
        finish_order = []
        
        # Play tricks until only one player remains
        while len(active_players) > 1:
            current_player_idx = self.play_trick(active_players, current_player_idx)
            
            # Remove players who finished
            for player in active_players.copy():
                if player.hand.is_empty():
                    active_players.remove(player)
                    finish_order.append(player)
                    player.finished_position = len(finish_order)
                    if current_player_idx >= len(active_players):
                        current_player_idx = 0
        
        # Last player finishes last
        if active_players:
            finish_order.append(active_players[0])
            active_players[0].finished_position = len(finish_order)
        
        # Assign roles and points
        self.assign_roles(finish_order)
    
    def play_trick(self, active_players: List[Player], start_idx: int) -> int:
        """Play a single trick and return next starting player index."""
        self.table.reset_passes(active_players)
        current_idx = start_idx
        passes_in_a_row = 0
        
        while True:
            player = active_players[current_idx]
            
            if not player.passed_this_trick:
                self.show_game_state(player, active_players)
                
                # Special case: first play must include 3 of clubs if round 1
                if self.round_num == 1 and self.table.current_combo is None and player.hand.has_three_of_clubs():
                    legal_combos = player.hand.get_legal_combos(None)
                    # Filter to only combos containing 3 of clubs
                    three_clubs = Card(Rank.THREE, Suit.CLUBS)
                    legal_combos = [c for c in legal_combos if three_clubs in c.cards]
                    
                    if isinstance(player, HumanPlayer):
                        print(f"\n{player.name}, you must play the 3♣ in your opening combo!")
                
                combo = player.choose_play(self.table.current_combo, self.options)
                
                if combo:
                    print(f"{player.name} plays: {combo}")
                    player.hand.remove_cards(list(combo.cards))
                    
                    # Check for immediate clear conditions
                    clears = self.table.play_combo(player, combo)
                    
                    # Check variant rules
                    if self.options.get("twos_clear", False) and combo.rank == Rank.TWO:
                        clears = True
                    elif (self.options.get("burn_on_equal", False) and 
                          self.table.current_combo and combo.rank == self.table.current_combo.rank):
                        clears = True
                    
                    if clears:
                        print(f"💥 Pile clears! {player.name} leads again.")
                        winner = self.table.clear_trick()
                        return active_players.index(winner)
                    
                    passes_in_a_row = 0
                else:
                    print(f"{player.name} passes.")
                    self.table.player_passes(player)
                    passes_in_a_row += 1
            else:
                passes_in_a_row += 1
            
            # Check if trick is over (all but one passed)
            active_in_trick = sum(1 for p in active_players if not p.passed_this_trick)
            if active_in_trick <= 1:
                winner = self.table.clear_trick()
                return active_players.index(winner)
            
            current_idx = (current_idx + 1) % len(active_players)
    
    def conduct_trading(self) -> None:
        """Handle inter-round trading between roles."""
        president = next((p for p in self.players if p.role == Role.PRESIDENT), None)
        asshole = next((p for p in self.players if p.role == Role.ASSHOLE), None)
        vp = next((p for p in self.players if p.role == Role.VICE_PRESIDENT), None)
        va = next((p for p in self.players if p.role == Role.VICE_ASSHOLE), None)
        
        if president and asshole:
            print(f"\n🔄 Trading: {president.name} ↔ {asshole.name}")
            
            # Asshole gives 2 highest cards to President
            cards_to_give = asshole.hand.get_highest_cards(2)
            asshole.hand.remove_cards(cards_to_give)
            president.hand.add_cards(cards_to_give)
            
            # President gives any 2 cards back
            cards_to_give_back = president.choose_cards_to_give(2)
            president.hand.remove_cards(cards_to_give_back)
            asshole.hand.add_cards(cards_to_give_back)
        
        if vp and va:
            print(f"🔄 Trading: {vp.name} ↔ {va.name}")
            
            # Vice Asshole gives 1 highest card to VP
            cards_to_give = va.hand.get_highest_cards(1)
            va.hand.remove_cards(cards_to_give)
            vp.hand.add_cards(cards_to_give)
            
            # VP gives any 1 card back
            cards_to_give_back = vp.choose_cards_to_give(1)
            vp.hand.remove_cards(cards_to_give_back)
            va.hand.add_cards(cards_to_give_back)
    
    def assign_roles(self, finish_order: List[Player]) -> None:
        """Assign roles based on finishing order."""
        roles = [Role.PRESIDENT, Role.VICE_PRESIDENT, Role.NEUTRAL, 
                Role.VICE_ASSHOLE, Role.ASSHOLE]
        
        # Pad neutral roles for games with more than 5 players
        while len(roles) < len(finish_order):
            roles.insert(-2, Role.NEUTRAL)
        
        for i, player in enumerate(finish_order):
            player.role = roles[i]
            player.total_points += player.role.value
    
    def show_game_state(self, current_player: Player, active_players: List[Player]) -> None:
        """Display current game state."""
        print(f"\n--- {current_player.name}'s Turn ---")
        
        if self.table.current_combo:
            print(f"Current pile: {self.table.current_combo} (played by {self.table.last_player.name})")
        else:
            print("Empty pile - you lead!")
        
        print("Players:")
        for player in active_players:
            status = f"{player.name} ({len(player.hand)} cards)"
            if player.role != Role.NEUTRAL:
                status += f" [{player.role.name}]"
            if player.passed_this_trick:
                status += " [PASSED]"
            print(f"  {status}")
    
    def show_round_summary(self) -> None:
        """Show round results and updated standings."""
        print(f"\n📊 Round {self.round_num} Results:")
        print("-" * 40)
        
        finish_order = sorted(self.players, key=lambda p: p.finished_position or 999)
        for i, player in enumerate(finish_order, 1):
            points_gained = player.role.value
            print(f"{i}. {player.name} - {player.role.name} "
                  f"({points_gained:+d} pts, total: {player.total_points})")
        
        print("\nCurrent Standings:")
        standings = sorted(self.players, key=lambda p: p.total_points, reverse=True)
        for i, player in enumerate(standings, 1):
            print(f"{i}. {player.name}: {player.total_points} points")
    
    def show_final_results(self) -> None:
        """Show final game results."""
        print(f"\n🏆 FINAL RESULTS")
        print("=" * 50)
        
        standings = sorted(self.players, key=lambda p: p.total_points, reverse=True)
        for i, player in enumerate(standings, 1):
            print(f"{i}. {player.name}: {player.total_points} points")
        
        winner = standings[0]
        print(f"\n👑 {winner.name} wins the game!")
    
    def save_game(self, filename: str) -> None:
        """Save game state to JSON file."""
        game_state = {
            "round_num": self.round_num,
            "players": [
                {
                    "name": p.name,
                    "type": "human" if isinstance(p, HumanPlayer) else "ai",
                    "total_points": p.total_points,
                    "role": p.role.name
                }
                for p in self.players
            ],
            "options": self.options
        }
        
        with open(filename, 'w') as f:
            json.dump(game_state, f, indent=2)
        print(f"Game saved to {filename}")


def run_self_test() -> None:
    """Run unit tests for game components."""
    print("Running self-tests...")
    
    # Test Card and Combo
    card1 = Card(Rank.KING, Suit.HEARTS)
    card2 = Card(Rank.KING, Suit.SPADES)
    card3 = Card(Rank.ACE, Suit.CLUBS)
    
    combo1 = Combo.from_cards([card1, card2])
    combo2 = Combo.from_cards([card3])
    
    assert combo1 is not None, "Should create pair combo"
    assert combo2 is not None, "Should create single combo"
    assert combo2.beats(combo1) == False, "Different sizes shouldn't compare"
    
    combo3 = Combo.from_cards([Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS)])
    assert combo3.beats(combo1), "Pair of Aces should beat pair of Kings"
    
    # Test Hand analysis
    hand = Hand([
        Card(Rank.THREE, Suit.CLUBS),
        Card(Rank.THREE, Suit.HEARTS),
        Card(Rank.KING, Suit.SPADES),
        Card(Rank.ACE, Suit.DIAMONDS)
    ])
    
    analysis = hand.get_analysis()
    assert 2 in analysis, "Should detect pair"
    assert Rank.THREE in analysis[2], "Should detect pair of threes"
    
    # Test legal combo generation
    legal_combos = hand.get_legal_combos(None)
    assert len(legal_combos) > 0, "Should have legal combos when leading"
    
    pair_combo = Combo.from_cards([Card(Rank.KING, Suit.HEARTS), Card(Rank.KING, Suit.CLUBS)])
    legal_vs_pair = hand.get_legal_combos(pair_combo)
    # Should have no legal pairs that beat K-K
    legal_pairs = [c for c in legal_vs_pair if c.size == 2]
    assert len(legal_pairs) == 0, "No pairs should beat K-K in this hand"
    
    print("✅ All self-tests passed!")


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Play Presidents card game")
    parser.add_argument("--players", type=int, default=4, choices=range(3, 7),
                       help="Number of players (3-6, default: 4)")
    parser.add_argument("--humans", type=int, default=1,
                       help="Number of human players (default: 1)")
    parser.add_argument("--rounds", type=int, default=5,
                       help="Number of rounds to play (default: 5)")
    parser.add_argument("--target-points", type=int,
                       help="End game when someone reaches this score")
    parser.add_argument("--burn-on-equal", action="store_true",
                       help="Equal rank plays clear the pile")
    parser.add_argument("--twos-clear", action="store_true",
                       help="Playing a 2 immediately clears the pile")
    parser.add_argument("--no-trading", action="store_true",
                       help="Disable inter-round trading")
    parser.add_argument("--ai-difficulty", choices=["easy", "medium"], default="medium",
                       help="AI difficulty level (default: medium)")
    parser.add_argument("--seed", type=int,
                       help="Random seed for deterministic games")
    parser.add_argument("--save", type=str,
                       help="Save game to JSON file")
    parser.add_argument("--load", type=str,
                       help="Load game from JSON file")
    parser.add_argument("--self-test", action="store_true",
                       help="Run unit tests and exit")
    
    args = parser.parse_args()
    
    if args.self_test:
        run_self_test()
        return
    
    # Validate arguments
    if args.humans > args.players:
        print("Error: Number of humans cannot exceed total players")
        return
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Set up game options
    options = {
        "rounds": args.rounds,
        "target_points": args.target_points,
        "burn_on_equal": args.burn_on_equal,
        "twos_clear": args.twos_clear,
        "trading": not args.no_trading,
        "ai_difficulty": args.ai_difficulty,
        "start_with_lowest": True  # Always enabled after round 1
    }
    
    try:
        if args.load:
            # Load game functionality would go here
            print(f"Loading from {args.load} not yet implemented")
            return
        
        # Create and run game
        game = Game(args.players, args.humans, **options)
        game.play_game()
        
        if args.save:
            game.save_game(args.save)
            
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user. Thanks for playing!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()