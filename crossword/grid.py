"""
Grid module for crossword puzzle generation.
Handles grid structure, symmetry, numbering, and slot management.
"""

import logging
from enum import Enum
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
import copy


class GridSymmetry(Enum):
    """Grid symmetry types for crossword puzzles."""
    ROTATIONAL_180 = "rotational_180"
    BARRED = "barred"
    ASYMMETRIC = "asymmetric"
    
    @classmethod
    def from_string(cls, s: str) -> 'GridSymmetry':
        """Create GridSymmetry from string."""
        for sym in cls:
            if sym.value == s.lower():
                return sym
        raise ValueError(f"Invalid symmetry: {s}")


class Direction(Enum):
    """Word direction in the grid."""
    ACROSS = "across"
    DOWN = "down"
    
    def __str__(self):
        return self.value


@dataclass
class GridSlot:
    """Represents a slot (sequence of white squares) in the grid."""
    start_row: int
    start_col: int
    direction: Direction
    length: int
    number: Optional[int] = None
    word: Optional[str] = None
    
    @property
    def coordinates(self) -> List[Tuple[int, int]]:
        """Get all coordinates occupied by this slot."""
        coords = []
        for i in range(self.length):
            if self.direction == Direction.ACROSS:
                coords.append((self.start_row, self.start_col + i))
            else:
                coords.append((self.start_row + i, self.start_col))
        return coords
    
    @property
    def end_row(self) -> int:
        """Get the ending row."""
        if self.direction == Direction.DOWN:
            return self.start_row + self.length - 1
        return self.start_row
    
    @property
    def end_col(self) -> int:
        """Get the ending column."""
        if self.direction == Direction.ACROSS:
            return self.start_col + self.length - 1
        return self.start_col
    
    def intersects_with(self, other: 'GridSlot') -> Optional[Tuple[int, int]]:
        """Check if this slot intersects with another slot."""
        for coord in self.coordinates:
            if coord in other.coordinates:
                return coord
        return None
    
    def get_crossing_position(self, other: 'GridSlot') -> Optional[Tuple[int, int]]:
        """Get the crossing position indices for both slots."""
        intersection = self.intersects_with(other)
        if not intersection:
            return None
        
        row, col = intersection
        
        # Calculate position within this slot
        if self.direction == Direction.ACROSS:
            my_pos = col - self.start_col
        else:
            my_pos = row - self.start_row
        
        # Calculate position within other slot
        if other.direction == Direction.ACROSS:
            other_pos = col - other.start_col
        else:
            other_pos = row - other.start_row
        
        return my_pos, other_pos


@dataclass
class ThemeEntry:
    """Represents a theme entry that must be placed in the grid."""
    word: str
    row: int
    col: int
    direction: Direction
    priority: int = 1  # Higher priority entries are placed first


class CrosswordGrid:
    """Represents a crossword puzzle grid with numbering and slot management."""
    
    # Grid cell states
    EMPTY = 0    # White square, unfilled
    FILLED = 1   # White square, filled with letter
    BLACK = 2    # Black square
    
    def __init__(self, width: int, height: int, symmetry: GridSymmetry, 
                 min_word_length: int = 3):
        """Initialize a crossword grid.
        
        Args:
            width: Grid width
            height: Grid height
            symmetry: Grid symmetry type
            min_word_length: Minimum word length allowed
        """
        self.width = width
        self.height = height
        self.symmetry = symmetry
        self.min_word_length = min_word_length
        
        # Grid structure: 2D array of cell states
        self.grid = [[self.EMPTY for _ in range(width)] for _ in range(height)]
        
        # Letter content: 2D array of letters (or None)
        self.letters = [[None for _ in range(width)] for _ in range(height)]
        
        # Slots (across and down)
        self.across_slots: List[GridSlot] = []
        self.down_slots: List[GridSlot] = []
        
        # Theme entries
        self.theme_entries: List[ThemeEntry] = []
        
        # Numbering
        self.numbers = [[None for _ in range(width)] for _ in range(height)]
        self.next_number = 1
        
        # Cache for slot lookups
        self._slot_cache: Dict[Tuple[int, int], List[GridSlot]] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize with basic symmetric pattern if needed
        if symmetry == GridSymmetry.ROTATIONAL_180:
            self._initialize_symmetric_pattern()
    
    def _initialize_symmetric_pattern(self):
        """Initialize a basic symmetric black square pattern."""
        # Start with a completely open grid - black squares will be added
        # during the solving process to maintain symmetry
        pass
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= row < self.height and 0 <= col < self.width
    
    def is_black_square(self, row: int, col: int) -> bool:
        """Check if position is a black square."""
        if not self.is_valid_position(row, col):
            return True  # Out of bounds treated as black
        return self.grid[row][col] == self.BLACK
    
    def is_white_square(self, row: int, col: int) -> bool:
        """Check if position is a white square."""
        if not self.is_valid_position(row, col):
            return False
        return self.grid[row][col] != self.BLACK
    
    def set_black_square(self, row: int, col: int, apply_symmetry: bool = True):
        """Set a square as black, optionally applying symmetry."""
        if not self.is_valid_position(row, col):
            return
        
        self.grid[row][col] = self.BLACK
        self.letters[row][col] = None
        
        # Apply symmetry if needed
        if apply_symmetry and self.symmetry == GridSymmetry.ROTATIONAL_180:
            sym_row = self.height - 1 - row
            sym_col = self.width - 1 - col
            if self.is_valid_position(sym_row, sym_col):
                self.grid[sym_row][sym_col] = self.BLACK
                self.letters[sym_row][sym_col] = None
    
    def set_letter(self, row: int, col: int, letter: str):
        """Set a letter in the grid."""
        if not self.is_valid_position(row, col) or self.is_black_square(row, col):
            return
        
        self.letters[row][col] = letter.upper()
        self.grid[row][col] = self.FILLED
    
    def get_letter(self, row: int, col: int) -> Optional[str]:
        """Get the letter at a position."""
        if not self.is_valid_position(row, col):
            return None
        return self.letters[row][col]
    
    def add_theme_entry(self, word: str, row: int, col: int, 
                       direction: str, priority: int = 1):
        """Add a theme entry that must be placed in the grid."""
        dir_enum = Direction.ACROSS if direction.lower() == 'across' else Direction.DOWN
        theme_entry = ThemeEntry(word.upper(), row, col, dir_enum, priority)
        
        # Validate theme entry placement
        if not self._validate_theme_entry(theme_entry):
            raise ValueError(f"Invalid theme entry placement: {word} at ({row},{col}) {direction}")
        
        self.theme_entries.append(theme_entry)
        
        # Sort by priority (higher priority first)
        self.theme_entries.sort(key=lambda x: x.priority, reverse=True)
    
    def _validate_theme_entry(self, entry: ThemeEntry) -> bool:
        """Validate that a theme entry can be placed."""
        # Check bounds
        if entry.direction == Direction.ACROSS:
            if entry.col + len(entry.word) > self.width:
                return False
        else:
            if entry.row + len(entry.word) > self.height:
                return False
        
        # Check for conflicts with existing theme entries
        for existing in self.theme_entries:
            if self._theme_entries_conflict(entry, existing):
                return False
        
        return True
    
    def _theme_entries_conflict(self, entry1: ThemeEntry, entry2: ThemeEntry) -> bool:
        """Check if two theme entries conflict."""
        # Create temporary slots to check intersection
        slot1 = GridSlot(entry1.row, entry1.col, entry1.direction, len(entry1.word))
        slot2 = GridSlot(entry2.row, entry2.col, entry2.direction, len(entry2.word))
        
        intersection = slot1.intersects_with(slot2)
        if not intersection:
            return False
        
        # If they intersect, check if the crossing letters match
        pos1, pos2 = slot1.get_crossing_position(slot2)
        return entry1.word[pos1] != entry2.word[pos2]
    
    def place_theme_entries(self):
        """Place all theme entries in the grid."""
        for entry in self.theme_entries:
            self._place_theme_entry(entry)
        
        # Regenerate slots after placing theme entries
        self._generate_slots()
        self._number_grid()
    
    def _place_theme_entry(self, entry: ThemeEntry):
        """Place a single theme entry in the grid."""
        for i, letter in enumerate(entry.word):
            if entry.direction == Direction.ACROSS:
                row, col = entry.row, entry.col + i
            else:
                row, col = entry.row + i, entry.col
            
            self.set_letter(row, col, letter)
    
    def _generate_slots(self):
        """Generate all valid slots (across and down) in the grid."""
        self.across_slots = []
        self.down_slots = []
        self._slot_cache = {}
        
        # Generate across slots
        for row in range(self.height):
            col = 0
            while col < self.width:
                if self.is_white_square(row, col):
                    # Start of potential slot
                    start_col = col
                    length = 0
                    
                    # Count consecutive white squares
                    while col < self.width and self.is_white_square(row, col):
                        length += 1
                        col += 1
                    
                    # Create slot if long enough
                    if length >= self.min_word_length:
                        slot = GridSlot(row, start_col, Direction.ACROSS, length)
                        self.across_slots.append(slot)
                        
                        # Update cache
                        for c in range(start_col, start_col + length):
                            key = (row, c)
                            if key not in self._slot_cache:
                                self._slot_cache[key] = []
                            self._slot_cache[key].append(slot)
                else:
                    col += 1
        
        # Generate down slots
        for col in range(self.width):
            row = 0
            while row < self.height:
                if self.is_white_square(row, col):
                    # Start of potential slot
                    start_row = row
                    length = 0
                    
                    # Count consecutive white squares
                    while row < self.height and self.is_white_square(row, col):
                        length += 1
                        row += 1
                    
                    # Create slot if long enough
                    if length >= self.min_word_length:
                        slot = GridSlot(start_row, col, Direction.DOWN, length)
                        self.down_slots.append(slot)
                        
                        # Update cache
                        for r in range(start_row, start_row + length):
                            key = (r, col)
                            if key not in self._slot_cache:
                                self._slot_cache[key] = []
                            self._slot_cache[key].append(slot)
                else:
                    row += 1
    
    def _number_grid(self):
        """Number the grid according to standard crossword numbering rules."""
        # Clear existing numbering
        self.numbers = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.next_number = 1
        
        # Number squares that start across or down words
        for row in range(self.height):
            for col in range(self.width):
                if self.is_white_square(row, col):
                    needs_number = False
                    
                    # Check if this starts an across word
                    if (col == 0 or self.is_black_square(row, col - 1)) and \
                       col + 1 < self.width and self.is_white_square(row, col + 1):
                        needs_number = True
                    
                    # Check if this starts a down word
                    if (row == 0 or self.is_black_square(row - 1, col)) and \
                       row + 1 < self.height and self.is_white_square(row + 1, col):
                        needs_number = True
                    
                    if needs_number:
                        self.numbers[row][col] = self.next_number
                        self.next_number += 1
        
        # Update slot numbers
        for slot in self.across_slots + self.down_slots:
            slot.number = self.numbers[slot.start_row][slot.start_col]
    
    def get_slots_at_position(self, row: int, col: int) -> List[GridSlot]:
        """Get all slots that pass through a given position."""
        return self._slot_cache.get((row, col), [])
    
    def get_crossing_slots(self, slot: GridSlot) -> List[Tuple[GridSlot, Tuple[int, int]]]:
        """Get all slots that cross with the given slot."""
        crossing_slots = []
        
        for coord in slot.coordinates:
            row, col = coord
            other_slots = self.get_slots_at_position(row, col)
            
            for other_slot in other_slots:
                if other_slot != slot and other_slot.direction != slot.direction:
                    crossing_pos = slot.get_crossing_position(other_slot)
                    if crossing_pos:
                        crossing_slots.append((other_slot, crossing_pos))
        
        return crossing_slots
    
    def can_place_word(self, slot: GridSlot, word: str) -> bool:
        """Check if a word can be placed in the given slot."""
        if len(word) != slot.length:
            return False
        
        # Check crossing constraints
        for other_slot, (my_pos, other_pos) in self.get_crossing_slots(slot):
            if other_slot.word:
                if word[my_pos] != other_slot.word[other_pos]:
                    return False
        
        return True
    
    def place_word(self, slot: GridSlot, word: str) -> bool:
        """Place a word in the given slot."""
        if not self.can_place_word(slot, word):
            return False
        
        slot.word = word.upper()
        
        # Update grid letters
        for i, letter in enumerate(word.upper()):
            if slot.direction == Direction.ACROSS:
                self.set_letter(slot.start_row, slot.start_col + i, letter)
            else:
                self.set_letter(slot.start_row + i, slot.start_col, letter)
        
        return True
    
    def remove_word(self, slot: GridSlot):
        """Remove a word from the given slot."""
        if not slot.word:
            return
        
        slot.word = None
        
        # Clear letters that are not part of crossing words
        for i in range(slot.length):
            if slot.direction == Direction.ACROSS:
                row, col = slot.start_row, slot.start_col + i
            else:
                row, col = slot.start_row + i, slot.start_col
            
            # Check if this position is used by other filled slots
            other_slots = self.get_slots_at_position(row, col)
            has_other_word = any(s.word and s != slot for s in other_slots)
            
            if not has_other_word:
                self.letters[row][col] = None
                self.grid[row][col] = self.EMPTY
    
    def get_all_slots(self) -> List[GridSlot]:
        """Get all slots in the grid."""
        return self.across_slots + self.down_slots
    
    def get_unfilled_slots(self) -> List[GridSlot]:
        """Get all unfilled slots in the grid."""
        return [slot for slot in self.get_all_slots() if not slot.word]
    
    def get_filled_slots(self) -> List[GridSlot]:
        """Get all filled slots in the grid."""
        return [slot for slot in self.get_all_slots() if slot.word]
    
    def is_complete(self) -> bool:
        """Check if the grid is completely filled."""
        return len(self.get_unfilled_slots()) == 0
    
    def get_fill_percentage(self) -> float:
        """Get the percentage of slots that are filled."""
        all_slots = self.get_all_slots()
        if not all_slots:
            return 0.0
        filled_slots = self.get_filled_slots()
        return len(filled_slots) / len(all_slots) * 100
    
    def validate_grid(self) -> List[str]:
        """Validate the grid and return a list of issues."""
        issues = []
        
        # Check for isolated regions
        if not self._is_grid_connected():
            issues.append("Grid contains isolated white square regions")
        
        # Check for two-letter words (if not allowed)
        for slot in self.get_all_slots():
            if slot.length < self.min_word_length:
                issues.append(f"Slot at ({slot.start_row},{slot.start_col}) {slot.direction} "
                            f"has length {slot.length} < minimum {self.min_word_length}")
        
        # Check crossing constraints
        for slot in self.get_filled_slots():
            for other_slot, (my_pos, other_pos) in self.get_crossing_slots(slot):
                if other_slot.word and slot.word[my_pos] != other_slot.word[other_pos]:
                    issues.append(f"Crossing conflict at ({slot.start_row + (my_pos if slot.direction == Direction.DOWN else 0)},"
                                f"{slot.start_col + (my_pos if slot.direction == Direction.ACROSS else 0)})")
        
        return issues
    
    def _is_grid_connected(self) -> bool:
        """Check if all white squares are connected (no isolated regions)."""
        # Find first white square
        start_pos = None
        for row in range(self.height):
            for col in range(self.width):
                if self.is_white_square(row, col):
                    start_pos = (row, col)
                    break
            if start_pos:
                break
        
        if not start_pos:
            return True  # No white squares, trivially connected
        
        # BFS to find all connected white squares
        visited = set()
        queue = [start_pos]
        visited.add(start_pos)
        
        while queue:
            row, col = queue.pop(0)
            
            # Check all 4 directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                
                if (new_row, new_col) not in visited and \
                   self.is_white_square(new_row, new_col):
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))
        
        # Count total white squares
        total_white = sum(1 for row in range(self.height) 
                         for col in range(self.width) 
                         if self.is_white_square(row, col))
        
        return len(visited) == total_white
    
    def get_black_square_count(self) -> int:
        """Get the total number of black squares."""
        return sum(1 for row in range(self.height) 
                  for col in range(self.width) 
                  if self.is_black_square(row, col))
    
    def get_white_square_count(self) -> int:
        """Get the total number of white squares."""
        return sum(1 for row in range(self.height) 
                  for col in range(self.width) 
                  if self.is_white_square(row, col))
    
    def get_word_count(self) -> Tuple[int, int]:
        """Get the count of across and down words."""
        return len(self.across_slots), len(self.down_slots)
    
    def clone(self) -> 'CrosswordGrid':
        """Create a deep copy of the grid."""
        new_grid = CrosswordGrid(self.width, self.height, self.symmetry, self.min_word_length)
        
        # Copy grid state
        new_grid.grid = [row[:] for row in self.grid]
        new_grid.letters = [row[:] for row in self.letters]
        new_grid.numbers = [row[:] for row in self.numbers]
        new_grid.next_number = self.next_number
        
        # Copy slots (deep copy)
        new_grid.across_slots = [copy.deepcopy(slot) for slot in self.across_slots]
        new_grid.down_slots = [copy.deepcopy(slot) for slot in self.down_slots]
        
        # Copy theme entries
        new_grid.theme_entries = [copy.deepcopy(entry) for entry in self.theme_entries]
        
        # Rebuild cache
        new_grid._slot_cache = {}
        for slot in new_grid.across_slots + new_grid.down_slots:
            for coord in slot.coordinates:
                if coord not in new_grid._slot_cache:
                    new_grid._slot_cache[coord] = []
                new_grid._slot_cache[coord].append(slot)
        
        return new_grid
    
    def to_dict(self) -> Dict:
        """Convert grid to dictionary representation."""
        return {
            'width': self.width,
            'height': self.height,
            'symmetry': self.symmetry.value,
            'min_word_length': self.min_word_length,
            'grid': self.grid,
            'letters': self.letters,
            'numbers': self.numbers,
            'across_slots': [
                {
                    'start_row': slot.start_row,
                    'start_col': slot.start_col,
                    'direction': slot.direction.value,
                    'length': slot.length,
                    'number': slot.number,
                    'word': slot.word
                }
                for slot in self.across_slots
            ],
            'down_slots': [
                {
                    'start_row': slot.start_row,
                    'start_col': slot.start_col,
                    'direction': slot.direction.value,
                    'length': slot.length,
                    'number': slot.number,
                    'word': slot.word
                }
                for slot in self.down_slots
            ],
            'theme_entries': [
                {
                    'word': entry.word,
                    'row': entry.row,
                    'col': entry.col,
                    'direction': entry.direction.value,
                    'priority': entry.priority
                }
                for entry in self.theme_entries
            ]
        }
    
    def __str__(self) -> str:
        """String representation of the grid."""
        lines = []
        
        for row in range(self.height):
            line = []
            for col in range(self.width):
                if self.is_black_square(row, col):
                    line.append('█')
                elif self.letters[row][col]:
                    line.append(self.letters[row][col])
                else:
                    line.append('·')
            lines.append(' '.join(line))
        
        return '\n'.join(lines)
    
    def print_with_numbers(self):
        """Print grid with numbers."""
        print("Grid with numbers:")
        for row in range(self.height):
            line = []
            for col in range(self.width):
                if self.is_black_square(row, col):
                    line.append('  █  ')
                else:
                    num = self.numbers[row][col]
                    letter = self.letters[row][col] or '·'
                    if num:
                        line.append(f"{num:2d}{letter} ")
                    else:
                        line.append(f"  {letter} ")
            print(''.join(line))
        
        print(f"\nSlots: {len(self.across_slots)} across, {len(self.down_slots)} down")
        print(f"Fill: {self.get_fill_percentage():.1f}%")


# Initialize grid and generate initial structure
def initialize_grid(width: int, height: int, symmetry: GridSymmetry, 
                   min_word_length: int = 3) -> CrosswordGrid:
    """Initialize and set up a crossword grid."""
    grid = CrosswordGrid(width, height, symmetry, min_word_length)
    
    # Place theme entries if any
    grid.place_theme_entries()
    
    # Generate initial slots and numbering
    grid._generate_slots()
    grid._number_grid()
    
    return grid
