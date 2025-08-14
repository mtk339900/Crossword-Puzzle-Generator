"""
Lexicon module for crossword puzzle generation.
Handles word loading, scoring, and management.
"""

import json
import csv
import logging
import math
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class WordEntry:
    """Represents a word in the lexicon with metadata."""
    word: str
    score: float
    frequency: int = 0
    part_of_speech: Optional[str] = None
    definition: Optional[str] = None
    is_proper_noun: bool = False
    is_abbreviation: bool = False
    is_plural: bool = False
    difficulty: str = "medium"  # easy, medium, hard
    categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        self.word = self.word.upper()


class Lexicon:
    """Manages a collection of words for crossword generation."""
    
    def __init__(self):
        """Initialize an empty lexicon."""
        self.words: Dict[str, WordEntry] = {}
        self.by_length: Dict[int, List[str]] = defaultdict(list)
        self.by_pattern: Dict[str, List[str]] = defaultdict(list)
        self.max_score = 0.0
        self.min_score = 100.0
        
        self.logger = logging.getLogger(__name__)
    
    def add_word(self, word_entry: WordEntry):
        """Add a word to the lexicon."""
        word = word_entry.word.upper()
        
        if word in self.words:
            # Update existing entry if new score is better
            if word_entry.score > self.words[word].score:
                self.words[word] = word_entry
        else:
            self.words[word] = word_entry
            self.by_length[len(word)].append(word)
        
        # Update score bounds
        self.max_score = max(self.max_score, word_entry.score)
        self.min_score = min(self.min_score, word_entry.score)
    
    def remove_word(self, word: str):
        """Remove a word from the lexicon."""
        word = word.upper()
        if word in self.words:
            del self.words[word]
            self.by_length[len(word)].remove(word)
    
    def remove_words(self, words: Set[str]):
        """Remove multiple words from the lexicon."""
        for word in words:
            self.remove_word(word)
    
    def has_word(self, word: str) -> bool:
        """Check if a word exists in the lexicon."""
        return word.upper() in self.words
    
    def get_word_entry(self, word: str) -> Optional[WordEntry]:
        """Get word entry by word."""
        return self.words.get(word.upper())
    
    def get_word_score(self, word: str) -> float:
        """Get the score for a word."""
        entry = self.get_word_entry(word)
        return entry.score if entry else 0.0
    
    def get_words_by_length(self, length: int) -> List[str]:
        """Get all words of a specific length."""
        return self.by_length.get(length, [])
    
    def get_words_by_pattern(self, pattern: str) -> List[str]:
        """Get words matching a pattern (. for any letter, letters for fixed)."""
        pattern = pattern.upper()
        
        # Check cache first
        if pattern in self.by_pattern:
            return self.by_pattern[pattern]
        
        # Generate matches
        regex_pattern = pattern.replace('.', '[A-Z]')
        regex = re.compile(f'^{regex_pattern}$')
        
        matching_words = []
        for word in self.get_words_by_length(len(pattern)):
            if regex.match(word):
                matching_words.append(word)
        
        # Sort by score (descending)
        matching_words.sort(key=lambda w: self.get_word_score(w), reverse=True)
        
        # Cache the result
        self.by_pattern[pattern] = matching_words
        
        return matching_words
    
    def get_words_with_constraints(self, length: int, fixed_letters: Dict[int, str], 
                                 min_score: float = 0.0) -> List[str]:
        """Get words with specific length and fixed letters at positions."""
        # Build pattern
        pattern = ['.'] * length
        for pos, letter in fixed_letters.items():
            if 0 <= pos < length:
                pattern[pos] = letter.upper()
        
        pattern_str = ''.join(pattern)
        words = self.get_words_by_pattern(pattern_str)
        
        # Filter by minimum score
        if min_score > 0:
            words = [w for w in words if self.get_word_score(w) >= min_score]
        
        return words
    
    def get_crossing_words(self, word: str, position: int) -> List[str]:
        """Get all words that can cross with given word at given position."""
        if not (0 <= position < len(word)):
            return []
        
        crossing_letter = word[position].upper()
        crossing_words = []
        
        # Look through all word lengths
        for length in self.by_length:
            if length < 3:  # Skip very short words
                continue
            
            for candidate in self.get_words_by_length(length):
                if crossing_letter in candidate:
                    crossing_words.append(candidate)
        
        return crossing_words
    
    def filter_by_difficulty(self, words: List[str], difficulty: str) -> List[str]:
        """Filter words by difficulty level."""
        if difficulty == "any":
            return words
        
        filtered = []
        for word in words:
            entry = self.get_word_entry(word)
            if entry and entry.difficulty == difficulty:
                filtered.append(word)
        
        return filtered
    
    def filter_by_categories(self, words: List[str], categories: List[str]) -> List[str]:
        """Filter words by categories."""
        if not categories:
            return words
        
        filtered = []
        for word in words:
            entry = self.get_word_entry(word)
            if entry and any(cat in entry.categories for cat in categories):
                filtered.append(word)
        
        return filtered
    
    def exclude_proper_nouns(self, words: List[str]) -> List[str]:
        """Exclude proper nouns from word list."""
        filtered = []
        for word in words:
            entry = self.get_word_entry(word)
            if not entry or not entry.is_proper_noun:
                filtered.append(word)
        
        return filtered
    
    def exclude_abbreviations(self, words: List[str]) -> List[str]:
        """Exclude abbreviations from word list."""
        filtered = []
        for word in words:
            entry = self.get_word_entry(word)
            if not entry or not entry.is_abbreviation:
                filtered.append(word)
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lexicon statistics."""
        stats = {
            'total_words': len(self.words),
            'by_length': {length: len(words) for length, words in self.by_length.items()},
            'score_range': (self.min_score, self.max_score),
            'proper_nouns': sum(1 for entry in self.words.values() if entry.is_proper_noun),
            'abbreviations': sum(1 for entry in self.words.values() if entry.is_abbreviation),
            'by_difficulty': defaultdict(int)
        }
        
        for entry in self.words.values():
            stats['by_difficulty'][entry.difficulty] += 1
        
        return stats
    
    def merge(self, other: 'Lexicon'):
        """Merge another lexicon into this one."""
        for word_entry in other.words.values():
            self.add_word(word_entry)
    
    def clear_cache(self):
        """Clear pattern matching cache."""
        self.by_pattern.clear()
    
    def __len__(self) -> int:
        return len(self.words)
    
    def __contains__(self, word: str) -> bool:
        return self.has_word(word)


class WordlistLoader:
    """Loads wordlists from various file formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_primary_wordlist(self, file_path: str) -> Lexicon:
        """Load the primary wordlist from a file."""
        return self.load_wordlist(file_path, is_primary=True)
    
    def load_wordlist(self, file_path: str, is_primary: bool = False) -> Lexicon:
        """Load a wordlist from a file (JSON, CSV, or TSV)."""
        lexicon = Lexicon()
        
        try:
            if file_path.lower().endswith('.json'):
                lexicon = self._load_json_wordlist(file_path)
            elif file_path.lower().endswith(('.csv', '.tsv')):
                lexicon = self._load_csv_wordlist(file_path)
            else:
                lexicon = self._load_text_wordlist(file_path)
            
            self.logger.info(f"Loaded {len(lexicon)} words from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading wordlist from {file_path}: {e}")
            if is_primary:
                # Create a minimal fallback lexicon
                lexicon = self._create_fallback_lexicon()
        
        return lexicon
    
    def _load_json_wordlist(self, file_path: str) -> Lexicon:
        """Load wordlist from JSON file."""
        lexicon = Lexicon()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Simple list of words
            for item in data:
                if isinstance(item, str):
                    word_entry = WordEntry(
                        word=item,
                        score=self._calculate_default_score(item)
                    )
                elif isinstance(item, dict):
                    word_entry = self._dict_to_word_entry(item)
                else:
                    continue
                
                lexicon.add_word(word_entry)
        
        elif isinstance(data, dict):
            # Dictionary format
            if 'words' in data:
                # Structured format
                for word_data in data['words']:
                    word_entry = self._dict_to_word_entry(word_data)
                    lexicon.add_word(word_entry)
            else:
                # Simple word -> score mapping
                for word, score in data.items():
                    word_entry = WordEntry(word=word, score=float(score))
                    lexicon.add_word(word_entry)
        
        return lexicon
    
    def _load_csv_wordlist(self, file_path: str) -> Lexicon:
        """Load wordlist from CSV/TSV file."""
        lexicon = Lexicon()
        
        # Detect delimiter
        delimiter = '\t' if file_path.lower().endswith('.tsv') else ','
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to detect if file has header
            sample = f.read(1024)
            f.seek(0)
            
            has_header = csv.Sniffer().has_header(sample)
            reader = csv.DictReader(f, delimiter=delimiter) if has_header else \
                     csv.reader(f, delimiter=delimiter)
            
            if has_header:
                for row in reader:
                    word_entry = self._dict_to_word_entry(row)
                    lexicon.add_word(word_entry)
            else:
                for row in reader:
                    if len(row) >= 1:
                        word = row[0].strip()
                        score = float(row[1]) if len(row) >= 2 and row[1] else \
                               self._calculate_default_score(word)
                        
                        word_entry = WordEntry(word=word, score=score)
                        lexicon.add_word(word_entry)
        
        return lexicon
    
    def _load_text_wordlist(self, file_path: str) -> Lexicon:
        """Load wordlist from plain text file (one word per line)."""
        lexicon = Lexicon()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):  # Skip comments
                    word_entry = WordEntry(
                        word=word,
                        score=self._calculate_default_score(word)
                    )
                    lexicon.add_word(word_entry)
        
        return lexicon
    
    def _dict_to_word_entry(self, data: Dict[str, Any]) -> WordEntry:
        """Convert dictionary data to WordEntry."""
        word = data.get('word', '').strip()
        if not word:
            raise ValueError("Word entry missing 'word' field")
        
        return WordEntry(
            word=word,
            score=float(data.get('score', self._calculate_default_score(word))),
            frequency=int(data.get('frequency', 0)),
            part_of_speech=data.get('part_of_speech') or data.get('pos'),
            definition=data.get('definition') or data.get('def'),
            is_proper_noun=bool(data.get('is_proper_noun', False)),
            is_abbreviation=bool(data.get('is_abbreviation', False)),
            is_plural=bool(data.get('is_plural', False)),
            difficulty=data.get('difficulty', 'medium'),
            categories=data.get('categories', []) if isinstance(data.get('categories'), list) 
                      else [data.get('categories')] if data.get('categories') else []
        )
    
    def _calculate_default_score(self, word: str) -> float:
        """Calculate a default score for a word based on various factors."""
        score = 50.0  # Base score
        
        # Length factor (prefer medium-length words)
        length = len(word)
        if 4 <= length <= 7:
            score += 10
        elif length >= 8:
            score += 5
        elif length == 3:
            score -= 10
        
        # Letter frequency factor
        common_letters = set('EAIOTRNSLUC')
        uncommon_letters = set('JQXZ')
        
        for letter in word:
            if letter in common_letters:
                score += 2
            elif letter in uncommon_letters:
                score -= 5
        
        # Vowel/consonant balance
        vowels = sum(1 for c in word if c in 'AEIOU')
        consonants = len(word) - vowels
        
        if vowels > 0 and consonants > 0:
            ratio = min(vowels, consonants) / max(vowels, consonants)
            score += ratio * 10
        
        # Avoid words with repeated letters (usually harder to cross)
        if len(set(word)) < len(word):
            score -= 5
        
        # Penalize words that look like abbreviations
        if word.isupper() and '.' not in word and len(word) <= 4:
            if any(char.isdigit() for char in word):
                score -= 20
        
        return max(0.0, min(100.0, score))
    
    def _create_fallback_lexicon(self) -> Lexicon:
        """Create a minimal fallback lexicon with basic words."""
        lexicon = Lexicon()
        
        # Basic high-scoring words for emergency use
        fallback_words = [
            ("THE", 90), ("AND", 85), ("FOR", 80), ("ARE", 80), ("BUT", 75),
            ("NOT", 75), ("YOU", 75), ("ALL", 75), ("CAN", 70), ("HER", 70),
            ("WAS", 70), ("ONE", 70), ("OUR", 70), ("OUT", 70), ("DAY", 70),
            ("GET", 65), ("HAS", 65), ("HIM", 65), ("HOW", 65), ("ITS", 65),
            ("MAY", 65), ("NEW", 65), ("NOW", 65), ("OLD", 65), ("SEE", 65),
            ("TWO", 65), ("WHO", 65), ("BOY", 60), ("DID", 60), ("MAN", 60),
            ("RUN", 60), ("SHE", 60), ("TOO", 60), ("USE", 60), ("WAY", 60)
        ]
        
        for word, score in fallback_words:
            word_entry = WordEntry(word=word, score=score)
            lexicon.add_word(word_entry)
        
        self.logger.warning("Using fallback lexicon with minimal word set")
        return lexicon
    
    def save_wordlist(self, lexicon: Lexicon, file_path: str, format_type: str = 'json'):
        """Save a lexicon to a file."""
        try:
            if format_type.lower() == 'json':
                self._save_json_wordlist(lexicon, file_path)
            elif format_type.lower() in ['csv', 'tsv']:
                delimiter = '\t' if format_type.lower() == 'tsv' else ','
                self._save_csv_wordlist(lexicon, file_path, delimiter)
            else:
                self._save_text_wordlist(lexicon, file_path)
            
            self.logger.info(f"Saved {len(lexicon)} words to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving wordlist to {file_path}: {e}")
    
    def _save_json_wordlist(self, lexicon: Lexicon, file_path: str):
        """Save lexicon to JSON file."""
        data = {
            'metadata': {
                'total_words': len(lexicon),
                'created_by': 'crossword-generator'
            },
            'words': []
        }
        
        for word_entry in lexicon.words.values():
            word_data = {
                'word': word_entry.word,
                'score': word_entry.score,
                'frequency': word_entry.frequency,
                'part_of_speech': word_entry.part_of_speech,
                'definition': word_entry.definition,
                'is_proper_noun': word_entry.is_proper_noun,
                'is_abbreviation': word_entry.is_abbreviation,
                'is_plural': word_entry.is_plural,
                'difficulty': word_entry.difficulty,
                'categories': word_entry.categories
            }
            data['words'].append(word_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_csv_wordlist(self, lexicon: Lexicon, file_path: str, delimiter: str = ','):
        """Save lexicon to CSV file."""
        fieldnames = ['word', 'score', 'frequency', 'part_of_speech', 'definition',
                     'is_proper_noun', 'is_abbreviation', 'is_plural', 'difficulty', 'categories']
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            
            for word_entry in lexicon.words.values():
                row = {
                    'word': word_entry.word,
                    'score': word_entry.score,
                    'frequency': word_entry.frequency,
                    'part_of_speech': word_entry.part_of_speech or '',
                    'definition': word_entry.definition or '',
                    'is_proper_noun': word_entry.is_proper_noun,
                    'is_abbreviation': word_entry.is_abbreviation,
                    'is_plural': word_entry.is_plural,
                    'difficulty': word_entry.difficulty,
                    'categories': ','.join(word_entry.categories) if word_entry.categories else ''
                }
                writer.writerow(row)
    
    def _save_text_wordlist(self, lexicon: Lexicon, file_path: str):
        """Save lexicon to plain text file."""
        words = sorted(lexicon.words.keys())
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for word in words:
                f.write(f"{word}\n")


class WordScorer:
    """Advanced word scoring system."""
    
    def __init__(self):
        # Letter frequency in English (approximate)
        self.letter_frequencies = {
            'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 6.97,
            'N': 6.95, 'S': 6.28, 'H': 6.09, 'R': 5.99, 'D': 4.25,
            'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
            'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
            'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
        }
        
        # Common bigrams and trigrams (for crossword-friendly words)
        self.good_bigrams = {
            'TH', 'ER', 'ON', 'AN', 'RE', 'ED', 'ND', 'OU', 'EA', 'NI',
            'TO', 'IT', 'IS', 'OR', 'TI', 'AS', 'TE', 'ET', 'NG', 'OF'
        }
        
        self.bad_patterns = {
            'II', 'UU', 'XXX', 'ZZZ', 'QQQ'  # Patterns that make words hard to cross
        }
    
    def score_word(self, word: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Score a word based on multiple factors."""
        word = word.upper()
        base_score = 50.0
        
        # Length scoring
        length_score = self._score_length(word)
        
        # Letter frequency scoring
        frequency_score = self._score_letter_frequency(word)
        
        # Pattern scoring (crossword-friendliness)
        pattern_score = self._score_patterns(word)
        
        # Vowel/consonant balance
        balance_score = self._score_vowel_consonant_balance(word)
        
        # Context-specific scoring
        context_score = 0.0
        if context:
            context_score = self._score_context(word, context)
        
        # Combine scores
        total_score = (
            base_score +
            length_score * 0.2 +
            frequency_score * 0.3 +
            pattern_score * 0.25 +
            balance_score * 0.15 +
            context_score * 0.1
        )
        
        return max(0.0, min(100.0, total_score))
    
    def _score_length(self, word: str) -> float:
        """Score based on word length (crossword preference)."""
        length = len(word)
        
        if length == 3:
            return -15  # Three-letter words are often poor fill
        elif length == 4:
            return 5
        elif 5 <= length <= 7:
            return 15  # Sweet spot for crosswords
        elif 8 <= length <= 10:
            return 10
        elif length >= 11:
            return 0   # Very long words can be hard to place
        else:
            return -20  # Very short words
    
    def _score_letter_frequency(self, word: str) -> float:
        """Score based on letter frequency (prefer common letters)."""
        total_frequency = sum(self.letter_frequencies.get(c, 0.1) for c in word)
        avg_frequency = total_frequency / len(word)
        
        # Normalize to 0-20 range
        return min(20, avg_frequency * 2)
    
    def _score_patterns(self, word: str) -> float:
        """Score based on letter patterns (crossword-friendliness)."""
        score = 0.0
        
        # Check for good bigrams
        for i in range(len(word) - 1):
            bigram = word[i:i+2]
            if bigram in self.good_bigrams:
                score += 2
        
        # Penalize bad patterns
        for pattern in self.bad_patterns:
            if pattern in word:
                score -= 10
        
        # Penalize repeated letters (harder to cross)
        unique_letters = len(set(word))
        if unique_letters < len(word):
            score -= (len(word) - unique_letters) * 3
        
        # Bonus for alternating vowels and consonants
        vowels = set('AEIOU')
        alternating_bonus = 0
        for i in range(len(word) - 1):
            curr_is_vowel = word[i] in vowels
            next_is_vowel = word[i+1] in vowels
            if curr_is_vowel != next_is_vowel:
                alternating_bonus += 1
        
        score += alternating_bonus * 0.5
        
        return score
    
    def _score_vowel_consonant_balance(self, word: str) -> float:
        """Score based on vowel/consonant balance."""
        vowels = sum(1 for c in word if c in 'AEIOU')
        consonants = len(word) - vowels
        
        if vowels == 0 or consonants == 0:
            return -10  # All vowels or all consonants is bad
        
        # Ideal ratio is around 0.4-0.6 vowels
        vowel_ratio = vowels / len(word)
        
        if 0.3 <= vowel_ratio <= 0.7:
            return 10
        elif 0.2 <= vowel_ratio <= 0.8:
            return 5
        else:
            return -5
    
    def _score_context(self, word: str, context: Dict[str, Any]) -> float:
        """Score based on context (theme, difficulty, etc.)."""
        score = 0.0
        
        # Theme bonus
        if context.get('is_theme_word', False):
            score += 10
        
        # Difficulty adjustment
        target_difficulty = context.get('target_difficulty', 'medium')
        if target_difficulty == 'easy' and len(word) <= 6:
            score += 5
        elif target_difficulty == 'hard' and len(word) >= 7:
            score += 5
        
        # Category bonus
        preferred_categories = context.get('preferred_categories', [])
        word_categories = context.get('word_categories', [])
        if any(cat in preferred_categories for cat in word_categories):
            score += 8
        
        return score
    
    def bulk_score_words(self, words: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Score multiple words efficiently."""
        scores = {}
        for word in words:
            scores[word] = self.score_word(word, context)
        return scores


def create_sample_wordlist():
    """Create a sample wordlist for testing purposes."""
    lexicon = Lexicon()
    
    # High-quality crossword words
    high_quality_words = [
        ("AREA", 85), ("RATE", 80), ("OARE", 75), ("RITE", 78), ("TORE", 76),
        ("TEAR", 82), ("REAR", 79), ("NEAR", 84), ("HEAR", 83), ("BEAR", 81),
        ("DEAR", 80), ("YEAR", 86), ("SEAT", 85), ("HEAT", 82), ("NEAT", 78),
        ("BEAT", 80), ("MEAT", 77), ("FEAT", 75), ("PEAT", 72), ("TEAM", 84),
        ("BEAM", 82), ("SEAM", 78), ("REAM", 74), ("CREAM", 79), ("DREAM", 81),
        ("STREAM", 76), ("MOTHER", 88), ("FATHER", 87), ("SISTER", 85), ("BROTHER", 84),
        ("FAMILY", 89), ("FRIEND", 86), ("PEOPLE", 88), ("PERSON", 83), ("STUDENT", 82),
        ("TEACHER", 85), ("SCHOOL", 87), ("HOUSE", 86), ("WATER", 89), ("LIGHT", 84),
        ("NIGHT", 83), ("RIGHT", 85), ("MIGHT", 78), ("SIGHT", 80), ("FIGHT", 77),
        ("THOUGHT", 82), ("THROUGH", 79), ("THOUGH", 81), ("ENOUGH", 83), ("ROUGH", 76)
    ]
    
    for word, score in high_quality_words:
        word_entry = WordEntry(
            word=word,
            score=score,
            difficulty="medium" if 70 <= score <= 85 else "easy" if score > 85 else "hard"
        )
        lexicon.add_word(word_entry)
    
    return lexicon
