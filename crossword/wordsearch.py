"""
Word Search puzzle generation module.
Handles placement of words and generation of distractor letters.
"""

import random
import logging
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass
from enum import Enum
import string


class WordDirection(Enum):
    """Directions for word placement in word search."""
    NORTH = "N"
    NORTHEAST = "NE"
    EAST = "E"
    SOUTHEAST = "SE"
    SOUTH = "S"
    SOUTHWEST = "SW"
    WEST = "W"
    NORTHWEST = "NW"
    
    def get_delta(self) -> Tuple[int, int]:
        """Get the (row, col) delta for this direction."""
        deltas = {
            WordDirection.NORTH: (-1, 0),
            WordDirection.NORTHEAST: (-1, 1),
            WordDirection.EAST: (0, 1),
            WordDirection.SOUTHEAST: (1, 1),
            WordDirection.SOUTH: (1, 0),
            WordDirection.SOUTHWEST: (1, -1),
            WordDirection.WEST: (0, -1),
            WordDirection.NORTHWEST: (-1, -1)
        }
        return deltas[self]
    
    @classmethod
    def from_string(cls, s: str) -> 'WordDirection':
        """Create WordDirection from string."""
        for direction in cls:
            if direction.value == s.upper():
                return direction
        raise ValueError(f"Invalid direction: {s}")


@dataclass
class PlacedWord:
    """Represents a word placed in the word search grid."""
    word: str
    start_row: int
    start_col: int
    direction: WordDirection
    
    @property
    def end_row(self) -> int:
        """Get the ending row of the word."""
        delta_row, _ = self.direction.get_delta()
        return self.start_row + delta_row * (len(self.word) - 1)
    
    @property
    def end_col(self) -> int:
        """Get the ending column of the word."""
        _, delta_col = self.direction.get_delta()
        return self.start_col + delta_col * (len(self.word) - 1)
    
    def get_coordinates(self) -> List[Tuple[int, int]]:
        """Get all coordinates occupied by this word."""
        coords = []
        delta_row, delta_col = self.direction.get_delta()
        
        for i in range(len(self.word)):
            row = self.start_row + i * delta_row
            col = self.start_col + i * delta_col
            coords.append((row, col))
        
        return coords
    
    def get_letter_at_position(self, row: int, col: int) -> Optional[str]:
        """Get the letter at a specific position, if this word occupies it."""
        coords = self.get_coordinates()
        for i, (word_row, word_col) in enumerate(coords):
            if word_row == row and word_col == col:
                return self.word[i]
        return None


@dataclass
class WordSearchPuzzle:
    """Represents a complete word search puzzle."""
    width: int
    height: int
    grid: List[List[str]]
    placed_words: List[PlacedWord]
    word_list: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert puzzle to dictionary representation."""
        return {
            'width': self.width,
            'height': self.height,
            'grid': self.grid,
            'placed_words': [
                {
                    'word': pw.word,
                    'start_row': pw.start_row,
                    'start_col': pw.start_col,
                    'end_row': pw.end_row,
                    'end_col': pw.end_col,
                    'direction': pw.direction.value
                }
                for pw in self.placed_words
            ],
            'word_list': sorted(self.word_list)
        }


class LetterFrequencyModel:
    """Model for generating realistic distractor letters."""
    
    def __init__(self, language: str = 'english'):
        """Initialize with language-specific letter frequencies."""
        self.language = language
        
        # English letter frequencies (approximate percentages)
        if language.lower() == 'english':
            self.frequencies = {
                'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 6.97,
                'N': 6.95, 'S': 6.28, 'H': 6.09, 'R': 5.99, 'D': 4.25,
                'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
                'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
                'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
            }
        else:
            # Default to uniform distribution
            self.frequencies = {chr(ord('A') + i): 3.85 for i in range(26)}
        
        # Create weighted letter list for random selection
        self.weighted_letters = []
        for letter, freq in self.frequencies.items():
            count = int(freq * 10)  # Scale up for better resolution
            self.weighted_letters.extend([letter] * count)
    
    def generate_letter(self, context: Optional[List[str]] = None) -> str:
        """Generate a letter based on frequency model."""
        if context:
            # Adjust probabilities based on context (nearby letters in grid)
            return self._generate_contextual_letter(context)
        else:
            return random.choice(self.weighted_letters)
    
    def _generate_contextual_letter(self, context: List[str]) -> str:
        """Generate letter considering nearby letters for better realism."""
        # Count context letters
        context_counts = {}
        for letter in context:
            if letter and letter.isalpha():
                context_counts[letter.upper()] = context_counts.get(letter.upper(), 0) + 1
        
        # Adjust probabilities to avoid too many similar letters nearby
        adjusted_frequencies = self.frequencies.copy()
        
        for letter, count in context_counts.items():
            if count >= 2:  # If letter appears multiple times in context
                adjusted_frequencies[letter] *= 0.3  # Reduce probability
        
        # Create adjusted weighted list
        adjusted_letters = []
        for letter, freq in adjusted_frequencies.items():
            count = int(freq * 10)
            adjusted_letters.extend([letter] * max(1, count))
        
        return random.choice(adjusted_letters)


class WordSearchGenerator:
    """Generator for word search puzzles."""
    
    def __init__(self, width: int = 20, height: int = 20, 
                 directions: List[str] = None, allow_overlaps: bool = True,
                 min_overlap: int = 2, max_placement_attempts: int = 1000):
        """Initialize the word search generator.
        
        Args:
            width: Grid width
            height: Grid height
            directions: List of allowed directions (N, NE, E, SE, S, SW, W, NW)
            allow_overlaps: Whether words can overlap
            min_overlap: Minimum letters that must overlap when overlapping is allowed
            max_placement_attempts: Maximum attempts to place each word
        """
        self.width = width
        self.height = height
        self.allow_overlaps = allow_overlaps
        self.min_overlap = min_overlap
        self.max_placement_attempts = max_placement_attempts
        
        # Set default directions if not provided
        if directions is None:
            directions = ['E', 'S', 'SE', 'NE']  # Common directions
        
        self.directions = [WordDirection.from_string(d) for d in directions]
        
        self.letter_model = LetterFrequencyModel()
        self.logger = logging.getLogger(__name__)
    
    def generate(self, words: List[str]) -> Optional[WordSearchPuzzle]:
        """Generate a word search puzzle with the given words.
        
        Args:
            words: List of words to place in the puzzle
            
        Returns:
            WordSearchPuzzle if successful, None if unable to place all words
        """
        if not words:
            self.logger.warning("No words provided for word search generation")
            return None
        
        # Validate and prepare words
        clean_words = self._prepare_words(words)
        if not clean_words:
            self.logger.error("No valid words after preparation")
            return None
        
        self.logger.info(f"Generating word search with {len(clean_words)} words")
        
        # Initialize empty grid
        grid = [['' for _ in range(self.width)] for _ in range(self.height)]
        placed_words = []
        
        # Sort words by length (longest first) for better placement
        sorted_words = sorted(clean_words, key=len, reverse=True)
        
        # Place words
        for word in sorted_words:
            placed_word = self._place_word(grid, word, placed_words)
            if placed_word:
                placed_words.append(placed_word)
                self._update_grid(grid, placed_word)
            else:
                self.logger.warning(f"Could not place word: {word}")
        
        if not placed_words:
            self.logger.error("Could not place any words")
            return None
        
        # Fill empty spaces with distractor letters
        self._fill_distractors(grid, placed_words)
        
        # Create puzzle object
        puzzle = WordSearchPuzzle(
            width=self.width,
            height=self.height,
            grid=grid,
            placed_words=placed_words,
            word_list=[pw.word for pw in placed_words]
        )
        
        self.logger.info(f"Successfully generated word search with {len(placed_words)}/{len(clean_words)} words")
        
        return puzzle
    
    def _prepare_words(self, words: List[str]) -> List[str]:
        """Prepare and validate words for placement."""
        clean_words = []
        
        for word in words:
            # Clean word
            clean_word = word.strip().upper()
            clean_word = ''.join(c for c in clean_word if c.isalpha())
            
            # Validate word
            if len(clean_word) < 2:
                self.logger.warning(f"Word too short: {word}")
                continue
            
            if len(clean_word) > max(self.width, self.height):
                self.logger.warning(f"Word too long for grid: {word}")
                continue
            
            clean_words.append(clean_word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in clean_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        return unique_words
    
    def _place_word(self, grid: List[List[str]], word: str, 
                   existing_words: List[PlacedWord]) -> Optional[PlacedWord]:
        """Attempt to place a word in the grid."""
        
        for attempt in range(self.max_placement_attempts):
            # Choose random position and direction
            direction = random.choice(self.directions)
            start_row, start_col = self._get_random_start_position(word, direction)
            
            if start_row is None or start_col is None:
                continue
            
            # Create potential placement
            potential_placement = PlacedWord(word, start_row, start_col, direction)
            
            # Check if placement is valid
            if self._is_valid_placement(grid, potential_placement, existing_words):
                return potential_placement
        
        return None
    
    def _get_random_start_position(self, word: str, direction: WordDirection) -> Tuple[Optional[int], Optional[int]]:
        """Get a random valid start position for a word in the given direction."""
        delta_row, delta_col = direction.get_delta()
        
        # Calculate bounds for start position
        if delta_row < 0:  # Moving up
            min_row = len(word) - 1
            max_row = self.height - 1
        elif delta_row > 0:  # Moving down
            min_row = 0
            max_row = self.height - len(word)
        else:  # Not moving vertically
            min_row = 0
            max_row = self.height - 1
        
        if delta_col < 0:  # Moving left
            min_col = len(word) - 1
            max_col = self.width - 1
        elif delta_col > 0:  # Moving right
            min_col = 0
            max_col = self.width - len(word)
        else:  # Not moving horizontally
            min_col = 0
            max_col = self.width - 1
        
        # Check if valid range exists
        if min_row > max_row or min_col > max_col:
            return None, None
        
        # Return random position within valid range
        start_row = random.randint(min_row, max_row)
        start_col = random.randint(min_col, max_col)
        
        return start_row, start_col
    
    def _is_valid_placement(self, grid: List[List[str]], placement: PlacedWord, 
                          existing_words: List[PlacedWord]) -> bool:
        """Check if a word placement is valid."""
        coordinates = placement.get_coordinates()
        
        # Check bounds
        for row, col in coordinates:
            if not (0 <= row < self.height and 0 <= col < self.width):
                return False
        
        # Check conflicts with existing letters
        overlap_count = 0
        for i, (row, col) in enumerate(coordinates):
            existing_letter = grid[row][col]
            word_letter = placement.word[i]
            
            if existing_letter:
                if existing_letter != word_letter:
                    return False  # Conflicting letter
                else:
                    overlap_count += 1
        
        # Check overlap rules
        if self.allow_overlaps:
            # If overlapping, ensure minimum overlap
            if overlap_count > 0 and overlap_count < self.min_overlap:
                return False
        else:
            # No overlaps allowed
            if overlap_count > 0:
                return False
        
        return True
    
    def _update_grid(self, grid: List[List[str]], placement: PlacedWord):
        """Update the grid with a placed word."""
        coordinates = placement.get_coordinates()
        
        for i, (row, col) in enumerate(coordinates):
            grid[row][col] = placement.word[i]
    
    def _fill_distractors(self, grid: List[List[str]], placed_words: List[PlacedWord]):
        """Fill empty cells with distractor letters."""
        # Collect letters used in words for frequency analysis
        word_letters = []
        for word in placed_words:
            word_letters.extend(list(word.word))
        
        # Fill each empty cell
        for row in range(self.height):
            for col in range(self.width):
                if not grid[row][col]:  # Empty cell
                    # Get context (surrounding letters)
                    context = self._get_context_letters(grid, row, col)
                    
                    # Generate appropriate distractor letter
                    distractor = self.letter_model.generate_letter(context)
                    grid[row][col] = distractor
    
    def _get_context_letters(self, grid: List[List[str]], row: int, col: int) -> List[str]:
        """Get surrounding letters for context-aware distractor generation."""
        context = []
        
        # Check 8 surrounding cells
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                r, c = row + dr, col + dc
                if 0 <= r < self.height and 0 <= c < self.width and grid[r][c]:
                    context.append(grid[r][c])
        
        return context
    
    def generate_answer_key(self, puzzle: WordSearchPuzzle) -> List[List[str]]:
        """Generate an answer key showing only the placed words."""
        answer_grid = [['.' for _ in range(puzzle.width)] for _ in range(puzzle.height)]
        
        for placed_word in puzzle.placed_words:
            coordinates = placed_word.get_coordinates()
            for i, (row, col) in enumerate(coordinates):
                answer_grid[row][col] = placed_word.word[i]
        
        return answer_grid
    
    def validate_puzzle(self, puzzle: WordSearchPuzzle) -> Tuple[bool, List[str]]:
        """Validate a word search puzzle."""
        issues = []
        
        # Check grid dimensions
        if len(puzzle.grid) != puzzle.height:
            issues.append(f"Grid height mismatch: expected {puzzle.height}, got {len(puzzle.grid)}")
        
        for i, row in enumerate(puzzle.grid):
            if len(row) != puzzle.width:
                issues.append(f"Grid width mismatch at row {i}: expected {puzzle.width}, got {len(row)}")
        
        # Check that all letters are valid
        for row in puzzle.grid:
            for cell in row:
                if not cell or not cell.isalpha():
                    issues.append("Grid contains invalid characters")
                    break
        
        # Check word placements
        for placed_word in puzzle.placed_words:
            # Verify word is actually in the grid
            coordinates = placed_word.get_coordinates()
            actual_word = ""
            
            for row, col in coordinates:
                if 0 <= row < puzzle.height and 0 <= col < puzzle.width:
                    actual_word += puzzle.grid[row][col]
                else:
                    issues.append(f"Word {placed_word.word} extends outside grid bounds")
                    break
            
            if actual_word != placed_word.word:
                issues.append(f"Word {placed_word.word} not found at specified location")
        
        # Check for duplicate words
        word_set = set()
        for placed_word in puzzle.placed_words:
            if placed_word.word in word_set:
                issues.append(f"Duplicate word: {placed_word.word}")
            word_set.add(placed_word.word)
        
        return len(issues) == 0, issues
    
    def get_puzzle_statistics(self, puzzle: WordSearchPuzzle) -> Dict[str, Any]:
        """Get statistics about the generated puzzle."""
        # Calculate word placement density
        total_cells = puzzle.width * puzzle.height
        word_cells = sum(len(pw.word) for pw in puzzle.placed_words)
        
        # Count unique letters
        all_letters = set()
        for row in puzzle.grid:
            for cell in row:
                if cell:
                    all_letters.add(cell)
        
        # Direction analysis
        direction_counts = {}
        for placed_word in puzzle.placed_words:
            direction = placed_word.direction.value
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        # Word length analysis
        word_lengths = [len(pw.word) for pw in puzzle.placed_words]
        
        return {
            'grid_size': f"{puzzle.width}x{puzzle.height}",
            'total_words': len(puzzle.placed_words),
            'word_placement_density': word_cells / total_cells,
            'unique_letters_used': len(all_letters),
            'direction_distribution': direction_counts,
            'word_length_stats': {
                'min': min(word_lengths) if word_lengths else 0,
                'max': max(word_lengths) if word_lengths else 0,
                'avg': sum(word_lengths) / len(word_lengths) if word_lengths else 0
            },
            'successful_placements': len(puzzle.placed_words)
        }


def create_themed_word_search(theme: str, word_count: int = 15, **kwargs) -> Optional[WordSearchPuzzle]:
    """Create a word search puzzle with words related to a specific theme."""
    # This would typically connect to a word database or API
    # For demonstration, using simple theme-based word lists
    
    themed_words = {
        'animals': ['CAT', 'DOG', 'BIRD', 'FISH', 'HORSE', 'COW', 'PIG', 'SHEEP', 
                   'LION', 'TIGER', 'BEAR', 'WOLF', 'RABBIT', 'MOUSE', 'ELEPHANT'],
        'colors': ['RED', 'BLUE', 'GREEN', 'YELLOW', 'PURPLE', 'ORANGE', 'BLACK', 
                  'WHITE', 'PINK', 'BROWN', 'GRAY', 'VIOLET', 'INDIGO', 'CYAN', 'MAGENTA'],
        'foods': ['APPLE', 'BREAD', 'CHEESE', 'PIZZA', 'PASTA', 'RICE', 'MEAT', 
                 'FISH', 'CAKE', 'COOKIE', 'BANANA', 'ORANGE', 'GRAPE', 'BERRY', 'SOUP'],
        'sports': ['SOCCER', 'TENNIS', 'GOLF', 'SWIM', 'RUN', 'JUMP', 'THROW', 
                  'CATCH', 'KICK', 'HIT', 'RACE', 'GAME', 'TEAM', 'WIN', 'SCORE'],
        'nature': ['TREE', 'FLOWER', 'GRASS', 'RIVER', 'LAKE', 'MOUNTAIN', 'SKY', 
                  'SUN', 'MOON', 'STAR', 'CLOUD', 'RAIN', 'WIND', 'SNOW', 'FOREST']
    }
    
    theme_lower = theme.lower()
    if theme_lower not in themed_words:
        return None
    
    # Select words for the theme
    available_words = themed_words[theme_lower]
    selected_words = random.sample(available_words, min(word_count, len(available_words)))
    
    # Generate puzzle
    generator = WordSearchGenerator(**kwargs)
    return generator.generate(selected_words)


def solve_word_search(puzzle: WordSearchPuzzle, target_words: List[str]) -> Dict[str, Optional[PlacedWord]]:
    """Find specified words in a word search puzzle."""
    found_words = {}
    
    for target_word in target_words:
        target_word = target_word.upper()
        found_words[target_word] = None
        
        # Search in all directions
        for direction in WordDirection:
            delta_row, delta_col = direction.get_delta()
            
            # Search starting from each position
            for start_row in range(puzzle.height):
                for start_col in range(puzzle.width):
                    # Check if word can fit in this direction from this position
                    end_row = start_row + delta_row * (len(target_word) - 1)
                    end_col = start_col + delta_col * (len(target_word) - 1)
                    
                    if not (0 <= end_row < puzzle.height and 0 <= end_col < puzzle.width):
                        continue
                    
                    # Extract word in this direction
                    found_word = ""
                    for i in range(len(target_word)):
                        row = start_row + i * delta_row
                        col = start_col + i * delta_col
                        found_word += puzzle.grid[row][col]
                    
                    # Check if it matches target
                    if found_word == target_word:
                        found_words[target_word] = PlacedWord(
                            word=target_word,
                            start_row=start_row,
                            start_col=start_col,
                            direction=direction
                        )
                        break
                
                if found_words[target_word] is not None:
                    break
            
            if found_words[target_word] is not None:
                break
    
    return found_words
