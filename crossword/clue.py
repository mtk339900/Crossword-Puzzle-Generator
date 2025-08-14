"""
Clue generation module for crossword puzzles.
Handles clue database management, template-based cluing, and cryptic clues.
"""

import re
import random
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import json

from crossword.grid import CrosswordGrid, GridSlot, Direction
from crossword.lexicon import Lexicon, WordEntry


class ClueDifficulty(Enum):
    """Difficulty levels for clues."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    
    @classmethod
    def from_string(cls, s: str) -> 'ClueDifficulty':
        """Create ClueDifficulty from string."""
        for diff in cls:
            if diff.value == s.lower():
                return diff
        return cls.MEDIUM


class ClueType(Enum):
    """Types of clues."""
    DEFINITION = "definition"
    SYNONYM = "synonym"
    ANAGRAM = "anagram"
    CHARADE = "charade"          # Word built from parts
    CONTAINER = "container"       # One word inside another
    REVERSAL = "reversal"        # Word spelled backwards
    HOMOPHONE = "homophone"      # Sounds like another word
    ABBREVIATION = "abbreviation"
    FILL_IN_BLANK = "fill_in_blank"
    WORDPLAY = "wordplay"
    CRYPTIC = "cryptic"          # Full cryptic clue


@dataclass
class ClueEntry:
    """Represents a clue for a word."""
    word: str
    clue: str
    clue_type: ClueType
    difficulty: ClueDifficulty
    source: str = "generated"
    confidence: float = 1.0
    enumeration: Optional[str] = None  # e.g., "(3,4)" for cryptic clues
    
    def __post_init__(self):
        self.word = self.word.upper()
        
        # Generate enumeration if not provided
        if self.enumeration is None and len(self.word.split()) > 1:
            parts = self.word.split()
            lengths = [str(len(part)) for part in parts]
            self.enumeration = f"({','.join(lengths)})"


class ClueDatabase:
    """Database of clues for words."""
    
    def __init__(self):
        """Initialize empty clue database."""
        self.clues: Dict[str, List[ClueEntry]] = defaultdict(list)
        self.by_type: Dict[ClueType, List[ClueEntry]] = defaultdict(list)
        self.by_difficulty: Dict[ClueDifficulty, List[ClueEntry]] = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def add_clue(self, clue_entry: ClueEntry):
        """Add a clue to the database."""
        word = clue_entry.word.upper()
        self.clues[word].append(clue_entry)
        self.by_type[clue_entry.clue_type].append(clue_entry)
        self.by_difficulty[clue_entry.difficulty].append(clue_entry)
    
    def get_clues(self, word: str) -> List[ClueEntry]:
        """Get all clues for a word."""
        return self.clues.get(word.upper(), [])
    
    def get_best_clue(self, word: str, difficulty: ClueDifficulty = ClueDifficulty.MEDIUM,
                     preferred_types: Optional[List[ClueType]] = None) -> Optional[ClueEntry]:
        """Get the best clue for a word based on criteria."""
        word = word.upper()
        candidates = self.get_clues(word)
        
        if not candidates:
            return None
        
        # Filter by difficulty
        filtered = [c for c in candidates if c.difficulty == difficulty]
        if not filtered:
            filtered = candidates  # Fall back to any difficulty
        
        # Filter by preferred types
        if preferred_types:
            type_filtered = [c for c in filtered if c.clue_type in preferred_types]
            if type_filtered:
                filtered = type_filtered
        
        # Sort by confidence
        filtered.sort(key=lambda c: c.confidence, reverse=True)
        
        return filtered[0]
    
    def load_from_file(self, file_path: str):
        """Load clues from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'clues' in data:
                for clue_data in data['clues']:
                    clue_entry = ClueEntry(
                        word=clue_data['word'],
                        clue=clue_data['clue'],
                        clue_type=ClueType(clue_data.get('type', 'definition')),
                        difficulty=ClueDifficulty(clue_data.get('difficulty', 'medium')),
                        source=clue_data.get('source', 'file'),
                        confidence=float(clue_data.get('confidence', 1.0)),
                        enumeration=clue_data.get('enumeration')
                    )
                    self.add_clue(clue_entry)
            
            self.logger.info(f"Loaded {sum(len(clues) for clues in self.clues.values())} clues from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading clue database from {file_path}: {e}")
    
    def save_to_file(self, file_path: str):
        """Save clues to a JSON file."""
        try:
            all_clues = []
            for clue_list in self.clues.values():
                for clue in clue_list:
                    all_clues.append({
                        'word': clue.word,
                        'clue': clue.clue,
                        'type': clue.clue_type.value,
                        'difficulty': clue.difficulty.value,
                        'source': clue.source,
                        'confidence': clue.confidence,
                        'enumeration': clue.enumeration
                    })
            
            data = {
                'metadata': {
                    'total_clues': len(all_clues),
                    'unique_words': len(self.clues)
                },
                'clues': all_clues
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved {len(all_clues)} clues to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving clue database to {file_path}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get clue database statistics."""
        total_clues = sum(len(clues) for clues in self.clues.values())
        
        return {
            'total_clues': total_clues,
            'unique_words': len(self.clues),
            'by_difficulty': {diff.value: len(clues) for diff, clues in self.by_difficulty.items()},
            'by_type': {type_.value: len(clues) for type_, clues in self.by_type.items()},
            'coverage_stats': {
                'words_with_multiple_clues': sum(1 for clues in self.clues.values() if len(clues) > 1),
                'average_clues_per_word': total_clues / max(len(self.clues), 1)
            }
        }


class ClueTemplates:
    """Templates for generating clues automatically."""
    
    def __init__(self):
        """Initialize clue templates."""
        self.definition_templates = [
            "{definition}",
            "Kind of {category}",
            "Type of {category}",
            "{category}, for one",
            "{category} variety"
        ]
        
        self.synonym_templates = [
            "{synonym}",
            "Like {synonym}",
            "Similar to {synonym}",
            "{synonym}, essentially"
        ]
        
        self.anagram_templates = [
            "Mixed up {anagram_source}",
            "Confused {anagram_source}",
            "Scrambled {anagram_source}",
            "Twisted {anagram_source}",
            "Rearranged {anagram_source}",
            "Muddled {anagram_source}"
        ]
        
        self.container_templates = [
            "{outer_part} containing {inner_part}",
            "{inner_part} in {outer_part}",
            "{outer_part} holds {inner_part}",
            "{inner_part} within {outer_part}"
        ]
        
        self.reversal_templates = [
            "{word} backwards",
            "{word} turned around",
            "{word} in reverse",
            "Backward {word}"
        ]
        
        # Common abbreviation patterns
        self.abbreviation_patterns = {
            'ABOUT': ['re', 'c', 'ca'],
            'ACCOUNT': ['ac', 'acc'],
            'AFTER': ['aft'],
            'AMERICA': ['US', 'USA'],
            'AND': ['&'],
            'ANSWER': ['ans'],
            'APPROXIMATELY': ['c', 'ca'],
            'ARMY': ['TA'],
            'ARTICLE': ['a', 'an', 'the'],
            'BACHELOR': ['BA', 'B'],
            'BOOK': ['b', 'vol'],
            'BRITISH': ['BR', 'UK'],
            'COMPANY': ['co', 'corp'],
            'DOCTOR': ['dr', 'md', 'mb'],
            'EAST': ['E'],
            'ENGLISH': ['eng'],
            'EUROPEAN': ['EU'],
            'EXAMPLE': ['eg'],
            'FIRST': ['I', 'a'],
            'FRANCE': ['F', 'FR'],
            'GERMAN': ['G', 'GER', 'D'],
            'GOOD': ['g'],
            'HUNDRED': ['c', 'C'],
            'ISLAND': ['I', 'is'],
            'LARGE': ['L', 'lg'],
            'LEFT': ['L', 'port'],
            'LOVE': ['o'],
            'MARRIED': ['m'],
            'MILLION': ['m'],
            'MISTER': ['mr'],
            'NORTH': ['N'],
            'NOTE': ['n'],
            'NUMBER': ['no', 'n'],
            'OLD': ['o'],
            'POINT': ['pt', 'n', 's', 'e', 'w'],
            'QUIET': ['sh', 'p'],
            'RIGHT': ['r', 'rt'],
            'RIVER': ['r'],
            'ROAD': ['rd', 'st'],
            'SMALL': ['s', 'sm'],
            'SOLDIER': ['gi'],
            'SOUTH': ['s'],
            'STREET': ['st', 'rd'],
            'THOUSAND': ['k', 'g'],
            'TIME': ['t'],
            'VERY': ['v'],
            'WEST': ['w'],
            'YEAR': ['y', 'yr']
        }
    
    def generate_definition_clue(self, word: str, definition: str) -> str:
        """Generate a definition-based clue."""
        if not definition:
            return f"Word meaning {word.lower()}"
        
        template = random.choice(self.definition_templates)
        return template.format(definition=definition)
    
    def generate_anagram_clue(self, word: str, source_phrase: Optional[str] = None) -> Optional[str]:
        """Generate an anagram clue."""
        if not source_phrase:
            # Try to find a good anagram source
            source_phrase = self._find_anagram_source(word)
        
        if not source_phrase:
            return None
        
        template = random.choice(self.anagram_templates)
        return template.format(anagram_source=source_phrase)
    
    def _find_anagram_source(self, word: str) -> Optional[str]:
        """Find a good source phrase for anagram clues."""
        # This is a simplified implementation
        # In practice, you'd want a database of anagram sources
        
        word = word.upper().replace(' ', '')
        
        # Simple anagram sources (this would be much more extensive in practice)
        anagram_sources = {
            'ACTOR': 'CARTO',
            'BELOW': 'BOWEL',
            'CARED': 'CEDAR',
            'DANCE': 'CANED',
            'EARTH': 'HEART',
            'FRIEND': 'FINDER',
            'GARDEN': 'DANGER',
            'HEART': 'EARTH',
            'ITEM': 'TIME',
            'LATE': 'TALE',
            'MATE': 'TEAM',
            'NEAR': 'EARN',
            'OCEAN': 'CANOE',
            'PART': 'TRAP',
            'RATE': 'TEAR',
            'SEAT': 'EAST',
            'TALE': 'LATE',
            'TIME': 'ITEM',
            'TRAP': 'PART',
            'WEAR': 'WARE'
        }
        
        return anagram_sources.get(word)
    
    def generate_abbreviation_clue(self, word: str) -> Optional[str]:
        """Generate a clue for an abbreviation."""
        word = word.upper()
        
        # Look for known abbreviation patterns
        for full_word, abbrevs in self.abbreviation_patterns.items():
            if word in abbrevs:
                return f"{full_word}, briefly"
        
        # Generic abbreviation clues
        if len(word) <= 3 and word.isupper():
            return f"Brief {word.lower()}"
        
        return None
    
    def generate_charade_clue(self, word: str) -> Optional[str]:
        """Generate a charade (word-building) clue."""
        # This is a complex feature that would require extensive word part databases
        # Simplified implementation for demonstration
        
        if len(word) >= 6:
            # Try to split word into recognizable parts
            mid = len(word) // 2
            part1 = word[:mid]
            part2 = word[mid:]
            
            # This would need a proper word parts database
            return f"{part1.lower()} and {part2.lower()}"
        
        return None


class CrypticClueGenerator:
    """Generator for cryptic crossword clues."""
    
    def __init__(self):
        """Initialize cryptic clue generator."""
        self.indicator_words = {
            'anagram': ['confused', 'mixed', 'twisted', 'broken', 'mad', 'wild', 'strange',
                       'odd', 'messy', 'upset', 'disturbed', 'rearranged', 'reformed'],
            'reversal': ['back', 'returned', 'reversed', 'backward', 'turned', 'about'],
            'container': ['in', 'within', 'inside', 'holding', 'containing', 'around'],
            'hidden': ['in', 'within', 'some', 'part of', 'hidden in'],
            'homophone': ['sounds', 'heard', 'spoken', 'said', 'audibly']
        }
        
        self.cryptic_templates = [
            "{definition} from {wordplay}",
            "{wordplay} gives {definition}",
            "{definition} - {wordplay}",
            "{wordplay} for {definition}"
        ]
    
    def generate_cryptic_clue(self, word: str, definition: str) -> Optional[str]:
        """Generate a cryptic crossword clue."""
        # This is a simplified cryptic clue generator
        # Real cryptic clues require extensive linguistic knowledge
        
        wordplay_options = []
        
        # Try anagram wordplay
        anagram_clue = self._generate_anagram_wordplay(word)
        if anagram_clue:
            wordplay_options.append(anagram_clue)
        
        # Try container wordplay
        container_clue = self._generate_container_wordplay(word)
        if container_clue:
            wordplay_options.append(container_clue)
        
        # Try reversal wordplay
        reversal_clue = self._generate_reversal_wordplay(word)
        if reversal_clue:
            wordplay_options.append(reversal_clue)
        
        if not wordplay_options:
            return None
        
        wordplay = random.choice(wordplay_options)
        template = random.choice(self.cryptic_templates)
        
        return template.format(definition=definition, wordplay=wordplay)
    
    def _generate_anagram_wordplay(self, word: str) -> Optional[str]:
        """Generate anagram-based wordplay."""
        # Find anagram source
        source = self._find_cryptic_anagram_source(word)
        if not source:
            return None
        
        indicator = random.choice(self.indicator_words['anagram'])
        return f"{source} {indicator}"
    
    def _generate_container_wordplay(self, word: str) -> Optional[str]:
        """Generate container-based wordplay."""
        if len(word) < 4:
            return None
        
        # Try to find container structure
        for i in range(1, len(word) - 1):
            for j in range(i + 1, len(word)):
                inner = word[i:j]
                outer_start = word[:i]
                outer_end = word[j:]
                
                if len(inner) >= 2 and len(outer_start) >= 1 and len(outer_end) >= 1:
                    outer = outer_start + outer_end
                    indicator = random.choice(self.indicator_words['container'])
                    return f"{inner.lower()} {indicator} {outer.lower()}"
        
        return None
    
    def _generate_reversal_wordplay(self, word: str) -> Optional[str]:
        """Generate reversal-based wordplay."""
        reversed_word = word[::-1]
        
        # Check if reversed word has meaning (simplified check)
        if len(reversed_word) >= 3:
            indicator = random.choice(self.indicator_words['reversal'])
            return f"{reversed_word.lower()} {indicator}"
        
        return None
    
    def _find_cryptic_anagram_source(self, word: str) -> Optional[str]:
        """Find anagram source for cryptic clues."""
        # This would use a comprehensive anagram database
        # Simplified implementation
        cryptic_anagrams = {
            'ACTOR': 'CARTO',
            'ALERT': 'ALTER',
            'ANGEL': 'GLEAN',
            'BEAST': 'BEATS',
            'BREAD': 'BEARD',
            'CHEAP': 'PEACH',
            'DREAM': 'DERMA',
            'EARTH': 'HATER',
            'FACTS': 'CRAFT',
            'GREAT': 'GRATE',
            'HEART': 'HATER',
            'ITEMS': 'TIMES',
            'LEAST': 'STEAL',
            'MEALS': 'MALES',
            'NOTES': 'STONE',
            'OCEAN': 'CANOE',
            'PARTS': 'STRAP',
            'RATES': 'TEARS',
            'STEAM': 'MEATS',
            'TEAMS': 'MEATS',
            'TREES': 'STEER'
        }
        
        return cryptic_anagrams.get(word.upper())


class ClueGenerator:
    """Main clue generator that coordinates different cluing methods."""
    
    def __init__(self, difficulty: ClueDifficulty = ClueDifficulty.MEDIUM,
                 cryptic_mode: bool = False, allow_abbreviations: bool = True,
                 allow_proper_nouns: bool = True, clue_db_path: Optional[str] = None):
        """Initialize the clue generator.
        
        Args:
            difficulty: Target difficulty level
            cryptic_mode: Whether to generate cryptic clues
            allow_abbreviations: Whether to allow abbreviation clues
            allow_proper_nouns: Whether to allow proper noun clues
            clue_db_path: Path to clue database file
        """
        self.difficulty = difficulty
        self.cryptic_mode = cryptic_mode
        self.allow_abbreviations = allow_abbreviations
        self.allow_proper_nouns = allow_proper_nouns
        
        self.clue_db = ClueDatabase()
        self.templates = ClueTemplates()
        self.cryptic_generator = CrypticClueGenerator()
        
        self.logger = logging.getLogger(__name__)
        
        # Load clue database if provided
        if clue_db_path:
            self.clue_db.load_from_file(clue_db_path)
        else:
            self._initialize_basic_clues()
    
    def _initialize_basic_clues(self):
        """Initialize with basic clues for common words."""
        basic_clues = [
            ClueEntry("THE", "Article", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("AND", "Plus", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("ARE", "Exist", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("FOR", "In favor of", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("NOT", "Negative", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("YOU", "Second person", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("ALL", "Everyone", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("CAN", "Able to", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("HER", "Belonging to she", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("WAS", "Existed", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("ONE", "Single", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("OUR", "Belonging to us", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("OUT", "Away", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("DAY", "24 hours", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("GET", "Obtain", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("HAS", "Possesses", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("HIM", "Male pronoun", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("HOW", "In what way", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("MAN", "Adult male", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("NEW", "Fresh", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("NOW", "At present", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("OLD", "Aged", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("SEE", "View", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("TWO", "Pair", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("WAY", "Method", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("WHO", "Which person", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("BOY", "Young male", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("DID", "Performed", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("ITS", "Belonging to it", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("LET", "Allow", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("PUT", "Place", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("SAY", "Speak", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("SHE", "Female pronoun", ClueType.DEFINITION, ClueDifficulty.EASY),
            ClueEntry("TOO", "Also", ClueType.SYNONYM, ClueDifficulty.EASY),
            ClueEntry("USE", "Employ", ClueType.SYNONYM, ClueDifficulty.EASY),
            
            # Medium difficulty
            ClueEntry("AREA", "Region", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("BACK", "Rear", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("BEST", "Finest", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("BOTH", "The two", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("CAME", "Arrived", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("COME", "Arrive", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("COULD", "Was able to", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("EACH", "Every", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("EVEN", "Level", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("FIND", "Locate", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("FIRST", "Initial", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("GIVE", "Donate", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("GOOD", "Fine", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("GREAT", "Excellent", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("GROUP", "Collection", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("HAND", "Appendage", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("HERE", "This place", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("HIGH", "Tall", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("HOME", "Residence", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("KNOW", "Understand", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("LARGE", "Big", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("LAST", "Final", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("LEFT", "Departed", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("LIFE", "Existence", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("LONG", "Extended", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("LOOK", "Glance", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("MADE", "Created", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("MAKE", "Create", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("MANY", "Numerous", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("MUCH", "A lot", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("NAME", "Title", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("NEED", "Require", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("NEXT", "Following", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("ONLY", "Sole", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("OPEN", "Unlock", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("OVER", "Above", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("PART", "Portion", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("PLACE", "Location", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("POINT", "Spot", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("RIGHT", "Correct", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("SAID", "Spoke", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("SAME", "Identical", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("SEEM", "Appear", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("SHOW", "Display", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("SMALL", "Tiny", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("SUCH", "Like that", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("TAKE", "Grab", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("THAN", "Compared to", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("THAT", "Those", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("THEM", "Those people", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("THERE", "That place", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("THESE", "Those items", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("THEY", "Those people", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("THINK", "Ponder", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("THIS", "Present item", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("TIME", "Duration", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("TURN", "Rotate", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("USED", "Employed", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("VERY", "Extremely", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("WANT", "Desire", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("WATER", "H2O", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("WELL", "Good", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("WENT", "Departed", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("WERE", "Existed", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("WHAT", "Which thing", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("WHEN", "At what time", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("WHERE", "At what place", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("WHICH", "What one", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("WHILE", "During", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("WILL", "Shall", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("WITH", "Alongside", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("WORK", "Labor", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("WORLD", "Earth", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("WOULD", "Might", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
            ClueEntry("YEAR", "12 months", ClueType.DEFINITION, ClueDifficulty.MEDIUM),
            ClueEntry("YOUNG", "Youthful", ClueType.SYNONYM, ClueDifficulty.MEDIUM)
        ]
        
        for clue_entry in basic_clues:
            self.clue_db.add_clue(clue_entry)
    
    def generate_clues(self, grid: CrosswordGrid, lexicon: Lexicon) -> Dict[GridSlot, ClueEntry]:
        """Generate clues for all filled slots in the grid."""
        clues = {}
        
        filled_slots = grid.get_filled_slots()
        self.logger.info(f"Generating clues for {len(filled_slots)} filled slots...")
        
        for slot in filled_slots:
            clue = self.generate_clue(slot.word, lexicon.get_word_entry(slot.word))
            if clue:
                clues[slot] = clue
            else:
                # Fallback clue
                clues[slot] = ClueEntry(
                    word=slot.word,
                    clue=f"Word with {len(slot.word)} letters",
                    clue_type=ClueType.DEFINITION,
                    difficulty=self.difficulty,
                    source="fallback",
                    confidence=0.1
                )
        
        return clues
    
    def generate_clue(self, word: str, word_entry: Optional[WordEntry] = None) -> Optional[ClueEntry]:
        """Generate a single clue for a word."""
        word = word.upper()
        
        # Try to get clue from database first
        db_clue = self.clue_db.get_best_clue(
            word, 
            self.difficulty,
            [ClueType.CRYPTIC] if self.cryptic_mode else None
        )
        
        if db_clue and db_clue.confidence >= 0.8:
            return db_clue
        
        # Generate new clue
        generated_clue = self._generate_new_clue(word, word_entry)
        
        # Return best available clue
        if generated_clue and (not db_clue or generated_clue.confidence > db_clue.confidence):
            return generated_clue
        
        return db_clue
    
    def _generate_new_clue(self, word: str, word_entry: Optional[WordEntry] = None) -> Optional[ClueEntry]:
        """Generate a new clue using various methods."""
        clue_candidates = []
        
        # Get word metadata
        definition = word_entry.definition if word_entry else None
        is_proper_noun = word_entry.is_proper_noun if word_entry else False
        is_abbreviation = word_entry.is_abbreviation if word_entry else False
        
        # Skip proper nouns if not allowed
        if is_proper_noun and not self.allow_proper_nouns:
            return None
        
        # Skip abbreviations if not allowed
        if is_abbreviation and not self.allow_abbreviations:
            return None
        
        # Try different clue generation methods
        
        # 1. Cryptic clues (if in cryptic mode)
        if self.cryptic_mode and definition:
            cryptic_clue = self.cryptic_generator.generate_cryptic_clue(word, definition)
            if cryptic_clue:
                clue_candidates.append(ClueEntry(
                    word=word,
                    clue=cryptic_clue,
                    clue_type=ClueType.CRYPTIC,
                    difficulty=self.difficulty,
                    source="generated_cryptic",
                    confidence=0.7
                ))
        
        # 2. Definition-based clues
        if definition:
            def_clue = self.templates.generate_definition_clue(word, definition)
            clue_candidates.append(ClueEntry(
                word=word,
                clue=def_clue,
                clue_type=ClueType.DEFINITION,
                difficulty=self.difficulty,
                source="generated_definition",
                confidence=0.8
            ))
        
        # 3. Abbreviation clues
        if is_abbreviation or (len(word) <= 4 and word.isupper()):
            abbrev_clue = self.templates.generate_abbreviation_clue(word)
            if abbrev_clue:
                clue_candidates.append(ClueEntry(
                    word=word,
                    clue=abbrev_clue,
                    clue_type=ClueType.ABBREVIATION,
                    difficulty=self.difficulty,
                    source="generated_abbreviation",
                    confidence=0.6
                ))
        
        # 4. Anagram clues
        if not self.cryptic_mode:  # Regular anagram clues for non-cryptic puzzles
            anagram_clue = self.templates.generate_anagram_clue(word)
            if anagram_clue:
                clue_candidates.append(ClueEntry(
                    word=word,
                    clue=anagram_clue,
                    clue_type=ClueType.ANAGRAM,
                    difficulty=self.difficulty,
                    source="generated_anagram",
                    confidence=0.5
                ))
        
        # 5. Charade clues
        charade_clue = self.templates.generate_charade_clue(word)
        if charade_clue:
            clue_candidates.append(ClueEntry(
                word=word,
                clue=charade_clue,
                clue_type=ClueType.CHARADE,
                difficulty=self.difficulty,
                source="generated_charade",
                confidence=0.4
            ))
        
        # Return best candidate
        if clue_candidates:
            clue_candidates.sort(key=lambda c: c.confidence, reverse=True)
            return clue_candidates[0]
        
        return None
    
    def validate_clue(self, word: str, clue: str) -> Tuple[bool, List[str]]:
        """Validate a clue for correctness and quality."""
        issues = []
        word = word.upper()
        
        # Basic validation
        if not clue.strip():
            issues.append("Clue is empty")
            return False, issues
        
        # Check if clue contains the answer
        clue_upper = clue.upper()
        if word in clue_upper:
            issues.append("Clue contains the answer")
        
        # Check for parts of the answer in the clue
        if len(word) > 4:
            for i in range(len(word) - 2):
                substr = word[i:i+3]
                if substr in clue_upper:
                    issues.append(f"Clue contains part of answer: '{substr}'")
        
        # Check clue length (reasonable bounds)
        if len(clue) < 3:
            issues.append("Clue is too short")
        elif len(clue) > 100:
            issues.append("Clue is too long")
        
        # Check for proper capitalization
        if not clue[0].isupper():
            issues.append("Clue should start with capital letter")
        
        # Grammar check (basic)
        if clue.endswith('.'):
            issues.append("Clue should not end with period")
        
        return len(issues) == 0, issues
    
    def get_clue_statistics(self) -> Dict[str, Any]:
        """Get statistics about clue generation."""
        db_stats = self.clue_db.get_statistics()
        
        return {
            'database_stats': db_stats,
            'generator_config': {
                'difficulty': self.difficulty.value,
                'cryptic_mode': self.cryptic_mode,
                'allow_abbreviations': self.allow_abbreviations,
                'allow_proper_nouns': self.allow_proper_nouns
            }
        }


def create_sample_clue_database() -> ClueDatabase:
    """Create a sample clue database for testing."""
    db = ClueDatabase()
    
    # Add some sample clues
    sample_clues = [
        ("AREA", "Region", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("RATE", "Speed", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("TIME", "Duration", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("TEAM", "Group", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("TEAR", "Rip", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("HEAR", "Listen", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("NEAR", "Close", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("BEAR", "Carry", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("DEAR", "Expensive", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("YEAR", "12 months", ClueType.DEFINITION, ClueDifficulty.EASY),
        ("WATER", "H2O", ClueType.DEFINITION, ClueDifficulty.EASY),
        ("LIGHT", "Illumination", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("NIGHT", "Dark time", ClueType.DEFINITION, ClueDifficulty.EASY),
        ("RIGHT", "Correct", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("MIGHT", "Power", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("SIGHT", "Vision", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
        ("FIGHT", "Battle", ClueType.SYNONYM, ClueDifficulty.MEDIUM),
    ]
    
    for word, clue, clue_type, difficulty in sample_clues:
        clue_entry = ClueEntry(
            word=word,
            clue=clue,
            clue_type=clue_type,
            difficulty=difficulty,
            source="sample",
            confidence=0.9
        )
        db.add_clue(clue_entry)
    
    return db
