"""
Quality analysis module for crossword puzzles.
Computes quality metrics and difficulty estimates.
"""

import logging
import math
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict

from crossword.grid import CrosswordGrid, Direction
from crossword.lexicon import Lexicon
from crossword.clue import ClueEntry, ClueDifficulty


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a crossword puzzle."""
    
    # Overall scores (0-100)
    overall_score: float = 0.0
    fill_score: float = 0.0
    difficulty_score: float = 0.0  # 1-10 scale
    theme_score: float = 0.0
    construction_score: float = 0.0
    
    # Detailed metrics
    word_scores: Dict[str, float] = field(default_factory=dict)
    average_word_score: float = 0.0
    word_count: int = 0
    
    # Grid metrics
    black_square_count: int = 0
    black_square_density: float = 0.0
    checked_square_ratio: float = 0.0
    open_square_count: int = 0
    
    # Fill quality
    three_letter_words: int = 0
    uncommon_words: int = 0
    abbreviations: int = 0
    proper_nouns: int = 0
    partials: int = 0
    crosswordese: int = 0
    
    # Letter analysis
    vowel_ratio: float = 0.0
    letter_distribution: Dict[str, int] = field(default_factory=dict)
    repeated_bigrams: int = 0
    
    # Advanced metrics
    pangram: bool = False
    cheater_squares: int = 0
    stacks: int = 0  # Number of word stacks
    connectivity_score: float = 0.0
    
    # Clue quality
    clue_variety_score: float = 0.0
    average_clue_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'overall_score': self.overall_score,
            'fill_score': self.fill_score,
            'difficulty_score': self.difficulty_score,
            'theme_score': self.theme_score,
            'construction_score': self.construction_score,
            'word_metrics': {
                'average_word_score': self.average_word_score,
                'word_count': self.word_count,
                'three_letter_words': self.three_letter_words,
                'uncommon_words': self.uncommon_words,
                'abbreviations': self.abbreviations,
                'proper_nouns': self.proper_nouns,
                'partials': self.partials,
                'crosswordese': self.crosswordese
            },
            'grid_metrics': {
                'black_square_count': self.black_square_count,
                'black_square_density': self.black_square_density,
                'checked_square_ratio': self.checked_square_ratio,
                'open_square_count': self.open_square_count,
                'cheater_squares': self.cheater_squares,
                'stacks': self.stacks,
                'connectivity_score': self.connectivity_score
            },
            'letter_analysis': {
                'vowel_ratio': self.vowel_ratio,
                'letter_distribution': self.letter_distribution,
                'repeated_bigrams': self.repeated_bigrams,
                'pangram': self.pangram
            },
            'clue_metrics': {
                'clue_variety_score': self.clue_variety_score,
                'average_clue_confidence': self.average_clue_confidence
            }
        }


class QualityAnalyzer:
    """Analyzer for crossword puzzle quality and difficulty."""
    
    def __init__(self):
        """Initialize the quality analyzer."""
        self.logger = logging.getLogger(__name__)
        
        # Common crosswordese (overused crossword words)
        self.crosswordese = {
            'ALOE', 'ANTE', 'AREA', 'ARIA', 'ERNE', 'ETUI', 'IDEA', 'ISLE',
            'OLEO', 'OLLA', 'OMIT', 'ORAL', 'ORCA', 'OREO', 'EARL', 'EELS',
            'ANOA', 'ANOW', 'ARIL', 'ARLES', 'ETAS', 'EWER', 'EWES', 'IBEX',
            'OAST', 'OATS', 'OCHS', 'ODEA', 'OPES', 'OPTS', 'UREA', 'UTES'
        }
        
        # Common partials (incomplete phrases)
        self.partial_patterns = {
            'PRE', 'UNI', 'TRI', 'EPI', 'SUB', 'MID', 'OUT', 'OFF', 'NON'
        }
        
        # Letter frequency for analysis
        self.vowels = set('AEIOU')
        self.consonants = set('BCDFGHJKLMNPQRSTVWXYZ')
    
    def analyze(self, grid: CrosswordGrid, lexicon: Lexicon, 
               clues: Optional[Dict[Any, ClueEntry]] = None) -> QualityMetrics:
        """Perform comprehensive quality analysis of a crossword puzzle."""
        
        self.logger.info("Starting quality analysis...")
        
        metrics = QualityMetrics()
        filled_slots = grid.get_filled_slots()
        
        if not filled_slots:
            self.logger.warning("No filled slots found for analysis")
            return metrics
        
        # Basic counts
        metrics.word_count = len(filled_slots)
        metrics.black_square_count = grid.get_black_square_count()
        total_squares = grid.width * grid.height
        metrics.black_square_density = metrics.black_square_count / total_squares
        
        # Analyze words and fill quality
        self._analyze_words(grid, lexicon, metrics)
        
        # Analyze grid structure
        self._analyze_grid_structure(grid, metrics)
        
        # Analyze letter distribution
        self._analyze_letters(grid, metrics)
        
        # Analyze clues if provided
        if clues:
            self._analyze_clues(clues, metrics)
        
        # Calculate composite scores
        self._calculate_composite_scores(metrics)
        
        self.logger.info(f"Quality analysis complete. Overall score: {metrics.overall_score:.1f}")
        
        return metrics
    
    def _analyze_words(self, grid: CrosswordGrid, lexicon: Lexicon, metrics: QualityMetrics):
        """Analyze word quality and fill characteristics."""
        filled_slots = grid.get_filled_slots()
        word_scores = []
        
        for slot in filled_slots:
            word = slot.word
            word_entry = lexicon.get_word_entry(word)
            
            # Get word score
            word_score = lexicon.get_word_score(word)
            word_scores.append(word_score)
            metrics.word_scores[word] = word_score
            
            # Count three-letter words
            if len(word) == 3:
                metrics.three_letter_words += 1
            
            # Count uncommon words (low scores)
            if word_score < 30:
                metrics.uncommon_words += 1
            
            # Count abbreviations and proper nouns
            if word_entry:
                if word_entry.is_abbreviation:
                    metrics.abbreviations += 1
                if word_entry.is_proper_noun:
                    metrics.proper_nouns += 1
            
            # Detect crosswordese
            if word in self.crosswordese:
                metrics.crosswordese += 1
            
            # Detect partials
            if any(word.startswith(pattern) for pattern in self.partial_patterns):
                if len(word) <= 5:  # Short words starting with common prefixes
                    metrics.partials += 1
        
        # Calculate average word score
        if word_scores:
            metrics.average_word_score = sum(word_scores) / len(word_scores)
    
    def _analyze_grid_structure(self, grid: CrosswordGrid, metrics: QualityMetrics):
        """Analyze grid structure and construction quality."""
        
        # Count checked squares (letters that are part of both across and down words)
        checked_squares = 0
        total_white_squares = 0
        
        for row in range(grid.height):
            for col in range(grid.width):
                if not grid.is_black_square(row, col):
                    total_white_squares += 1
                    slots_at_pos = grid.get_slots_at_position(row, col)
                    filled_slots_at_pos = [s for s in slots_at_pos if s.word]
                    
                    if len(filled_slots_at_pos) >= 2:
                        checked_squares += 1
        
        metrics.checked_square_ratio = checked_squares / max(total_white_squares, 1)
        
        # Count open squares (unchecked letters)
        metrics.open_square_count = total_white_squares - checked_squares
        
        # Detect cheater squares (black squares that don't affect word count)
        metrics.cheater_squares = self._count_cheater_squares(grid)
        
        # Count word stacks (parallel words)
        metrics.stacks = self._count_stacks(grid)
        
        # Calculate connectivity score
        metrics.connectivity_score = self._calculate_connectivity(grid)
    
    def _analyze_letters(self, grid: CrosswordGrid, metrics: QualityMetrics):
        """Analyze letter distribution and patterns."""
        all_letters = []
        letter_counts = Counter()
        
        # Collect all letters from filled words
        for slot in grid.get_filled_slots():
            for letter in slot.word:
                all_letters.append(letter)
                letter_counts[letter] += 1
        
        if not all_letters:
            return
        
        # Calculate vowel ratio
        vowel_count = sum(1 for letter in all_letters if letter in self.vowels)
        metrics.vowel_ratio = vowel_count / len(all_letters)
        
        # Store letter distribution
        metrics.letter_distribution = dict(letter_counts)
        
        # Check for pangram (all 26 letters used)
        metrics.pangram = len(set(all_letters)) == 26
        
        # Count repeated bigrams
        metrics.repeated_bigrams = self._count_repeated_bigrams(grid)
    
    def _analyze_clues(self, clues: Dict[Any, ClueEntry], metrics: QualityMetrics):
        """Analyze clue quality and variety."""
        if not clues:
            return
        
        clue_types = Counter()
        clue_difficulties = Counter()
        confidence_sum = 0
        
        for clue_entry in clues.values():
            clue_types[clue_entry.clue_type.value] += 1
            clue_difficulties[clue_entry.difficulty.value] += 1
            confidence_sum += clue_entry.confidence
        
        # Calculate variety score (higher is better for more variety)
        total_clues = len(clues)
        type_entropy = self._calculate_entropy(clue_types, total_clues)
        difficulty_entropy = self._calculate_entropy(clue_difficulties, total_clues)
        
        metrics.clue_variety_score = (type_entropy + difficulty_entropy) * 50
        metrics.average_clue_confidence = confidence_sum / total_clues
    
    def _calculate_composite_scores(self, metrics: QualityMetrics):
        """Calculate composite quality scores."""
        
        # Fill score (0-100)
        fill_components = []
        
        # Word quality component
        word_quality = min(100, metrics.average_word_score)
        fill_components.append(word_quality * 0.4)
        
        # Penalize poor fill
        penalties = 0
        penalties += metrics.three_letter_words * 5  # Penalize 3-letter words
        penalties += metrics.uncommon_words * 3
        penalties += metrics.crosswordese * 4
        penalties += metrics.partials * 6
        penalties += metrics.abbreviations * 2
        
        fill_penalty = min(40, penalties)
        fill_components.append(max(0, 60 - fill_penalty))
        
        metrics.fill_score = sum(fill_components)
        
        # Construction score (0-100)
        construction_components = []
        
        # Grid structure quality
        structure_score = 50
        structure_score += (metrics.checked_square_ratio - 0.75) * 100  # Target ~75% checked
        structure_score -= metrics.open_square_count * 2  # Penalize unchecked squares
        structure_score -= metrics.cheater_squares * 5  # Penalize cheater squares
        structure_score += metrics.stacks * 3  # Reward interesting construction
        structure_score += metrics.connectivity_score * 20
        
        construction_components.append(max(0, min(100, structure_score)))
        
        metrics.construction_score = sum(construction_components) / len(construction_components)
        
        # Theme score (simplified - would need theme analysis)
        metrics.theme_score = 50  # Neutral score without theme analysis
        
        # Difficulty score (1-10 scale)
        difficulty_factors = []
        difficulty_factors.append(min(10, max(1, metrics.average_word_score / 10)))
        difficulty_factors.append(min(10, max(1, 5 + metrics.uncommon_words * 0.5)))
        difficulty_factors.append(min(10, max(1, 3 + metrics.three_letter_words * 0.3)))
        
        if metrics.average_clue_confidence > 0:
            difficulty_factors.append(min(10, max(1, 10 - metrics.average_clue_confidence * 5)))
        
        metrics.difficulty_score = sum(difficulty_factors) / len(difficulty_factors)
        
        # Overall score (0-100)
        overall_components = [
            metrics.fill_score * 0.4,
            metrics.construction_score * 0.3,
            metrics.theme_score * 0.15,
            min(100, metrics.clue_variety_score) * 0.15
        ]
        
        metrics.overall_score = sum(overall_components)
    
    def _count_cheater_squares(self, grid: CrosswordGrid) -> int:
        """Count cheater squares (black squares that don't change word count)."""
        cheater_count = 0
        
        # This is a simplified detection - a full implementation would
        # test removing each black square to see if word count changes
        for row in range(1, grid.height - 1):
            for col in range(1, grid.width - 1):
                if grid.is_black_square(row, col):
                    # Check if surrounded by other black squares
                    neighbors_black = sum(1 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                                        if grid.is_black_square(row + dr, col + dc))
                    if neighbors_black >= 2:
                        cheater_count += 1
        
        return cheater_count
    
    def _count_stacks(self, grid: CrosswordGrid) -> int:
        """Count word stacks (parallel adjacent words)."""
        stacks = 0
        
        # Count horizontal stacks
        for row in range(grid.height - 1):
            consecutive_across = 0
            for col in range(grid.width):
                if (not grid.is_black_square(row, col) and 
                    not grid.is_black_square(row + 1, col)):
                    consecutive_across += 1
                else:
                    if consecutive_across >= 3:  # Stack of at least 3
                        stacks += 1
                    consecutive_across = 0
            
            if consecutive_across >= 3:
                stacks += 1
        
        # Count vertical stacks
        for col in range(grid.width - 1):
            consecutive_down = 0
            for row in range(grid.height):
                if (not grid.is_black_square(row, col) and 
                    not grid.is_black_square(row, col + 1)):
                    consecutive_down += 1
                else:
                    if consecutive_down >= 3:
                        stacks += 1
                    consecutive_down = 0
            
            if consecutive_down >= 3:
                stacks += 1
        
        return stacks
    
    def _calculate_connectivity(self, grid: CrosswordGrid) -> float:
        """Calculate how well-connected the grid is (0-1 scale)."""
        # This measures how many words cross other words
        total_crossings = 0
        total_possible = 0
        
        for slot in grid.get_filled_slots():
            crossing_slots = grid.get_crossing_slots(slot)
            actual_crossings = len([cs for cs, _ in crossing_slots if cs.word])
            total_crossings += actual_crossings
            total_possible += len(crossing_slots)
        
        if total_possible == 0:
            return 0.0
        
        return total_crossings / total_possible
    
    def _count_repeated_bigrams(self, grid: CrosswordGrid) -> int:
        """Count repeated two-letter combinations."""
        bigrams = Counter()
        
        for slot in grid.get_filled_slots():
            word = slot.word
            for i in range(len(word) - 1):
                bigram = word[i:i+2]
                bigrams[bigram] += 1
        
        # Count bigrams that appear more than twice
        repeated = sum(1 for count in bigrams.values() if count > 2)
        return repeated
    
    def _calculate_entropy(self, counter: Counter, total: int) -> float:
        """Calculate entropy for measuring variety."""
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def generate_quality_report(self, metrics: QualityMetrics) -> str:
        """Generate a human-readable quality report."""
        report = []
        
        report.append("CROSSWORD QUALITY ANALYSIS")
        report.append("=" * 30)
        report.append("")
        
        # Overall scores
        report.append("OVERALL SCORES:")
        report.append(f"  Overall Quality: {metrics.overall_score:.1f}/100")
        report.append(f"  Fill Quality: {metrics.fill_score:.1f}/100")
        report.append(f"  Construction: {metrics.construction_score:.1f}/100")
        report.append(f"  Difficulty: {metrics.difficulty_score:.1f}/10")
        report.append("")
        
        # Word analysis
        report.append("WORD ANALYSIS:")
        report.append(f"  Total Words: {metrics.word_count}")
        report.append(f"  Average Word Score: {metrics.average_word_score:.1f}")
        report.append(f"  Three-letter Words: {metrics.three_letter_words}")
        report.append(f"  Uncommon Words: {metrics.uncommon_words}")
        report.append(f"  Abbreviations: {metrics.abbreviations}")
        report.append(f"  Proper Nouns: {metrics.proper_nouns}")
        report.append(f"  Crosswordese: {metrics.crosswordese}")
        report.append(f"  Partials: {metrics.partials}")
        report.append("")
        
        # Grid analysis
        report.append("GRID ANALYSIS:")
        report.append(f"  Black Square Density: {metrics.black_square_density:.1%}")
        report.append(f"  Checked Square Ratio: {metrics.checked_square_ratio:.1%}")
        report.append(f"  Open Squares: {metrics.open_square_count}")
        report.append(f"  Cheater Squares: {metrics.cheater_squares}")
        report.append(f"  Word Stacks: {metrics.stacks}")
        report.append(f"  Connectivity: {metrics.connectivity_score:.2f}")
        report.append("")
        
        # Letter analysis
        report.append("LETTER ANALYSIS:")
        report.append(f"  Vowel Ratio: {metrics.vowel_ratio:.1%}")
        report.append(f"  Repeated Bigrams: {metrics.repeated_bigrams}")
        report.append(f"  Pangram: {'Yes' if metrics.pangram else 'No'}")
        report.append("")
        
        # Clue analysis (if available)
        if metrics.average_clue_confidence > 0:
            report.append("CLUE ANALYSIS:")
            report.append(f"  Clue Variety Score: {metrics.clue_variety_score:.1f}")
            report.append(f"  Average Confidence: {metrics.average_clue_confidence:.2f}")
            report.append("")
        
        # Quality assessment
        report.append("QUALITY ASSESSMENT:")
        if metrics.overall_score >= 80:
            assessment = "Excellent - Publication quality"
        elif metrics.overall_score >= 65:
            assessment = "Good - Minor improvements needed"
        elif metrics.overall_score >= 50:
            assessment = "Fair - Several issues to address"
        else:
            assessment = "Poor - Major reconstruction needed"
        
        report.append(f"  {assessment}")
        report.append("")
        
        # Recommendations
        recommendations = self._generate_recommendations(metrics)
        if recommendations:
            report.append("RECOMMENDATIONS:")
            for rec in recommendations:
                report.append(f"  • {rec}")
        
        return "\n".join(report)
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        if metrics.three_letter_words > metrics.word_count * 0.15:
            recommendations.append("Reduce number of three-letter words")
        
        if metrics.uncommon_words > metrics.word_count * 0.1:
            recommendations.append("Replace uncommon words with more accessible alternatives")
        
        if metrics.crosswordese > 3:
            recommendations.append("Minimize use of crosswordese (overused crossword words)")
        
        if metrics.partials > 2:
            recommendations.append("Avoid partial words and incomplete phrases")
        
        if metrics.checked_square_ratio < 0.7:
            recommendations.append("Increase checking - more letters should be part of both across and down words")
        
        if metrics.open_square_count > metrics.word_count * 0.2:
            recommendations.append("Reduce unchecked squares for better construction")
        
        if metrics.cheater_squares > 0:
            recommendations.append("Remove cheater squares that don't contribute to word count")
        
        if metrics.vowel_ratio < 0.3 or metrics.vowel_ratio > 0.5:
            recommendations.append("Improve vowel/consonant balance in word selection")
        
        if metrics.black_square_density > 0.25:
            recommendations.append("Consider reducing black square density for more open fill")
        
        if metrics.connectivity_score < 0.7:
            recommendations.append("Improve grid connectivity - words should cross more frequently")
        
        return recommendations


def analyze_puzzle_difficulty(grid: CrosswordGrid, lexicon: Lexicon, 
                            clues: Optional[Dict[Any, ClueEntry]] = None) -> Dict[str, Any]:
    """Standalone function to analyze puzzle difficulty."""
    analyzer = QualityAnalyzer()
    metrics = analyzer.analyze(grid, lexicon, clues)
    
    difficulty_analysis = {
        'difficulty_score': metrics.difficulty_score,
        'difficulty_level': 'Easy' if metrics.difficulty_score <= 3 else
                          'Medium' if metrics.difficulty_score <= 7 else 'Hard',
        'contributing_factors': {
            'word_complexity': metrics.average_word_score,
            'uncommon_words': metrics.uncommon_words,
            'abbreviations': metrics.abbreviations,
            'three_letter_words': metrics.three_letter_words,
            'grid_complexity': metrics.connectivity_score
        },
        'solver_experience_estimate': {
            'beginner': 'Suitable' if metrics.difficulty_score <= 4 else 'Too difficult',
            'intermediate': 'Suitable' if 3 <= metrics.difficulty_score <= 7 else 
                          'Too easy' if metrics.difficulty_score < 3 else 'Too difficult',
            'expert': 'Suitable' if metrics.difficulty_score >= 6 else 'Too easy'
        }
    }
    
    return difficulty_analysis


def compare_puzzles(metrics1: QualityMetrics, metrics2: QualityMetrics) -> Dict[str, Any]:
    """Compare two puzzles and provide analysis."""
    comparison = {
        'overall_winner': 'Puzzle 1' if metrics1.overall_score > metrics2.overall_score else 'Puzzle 2',
        'score_difference': abs(metrics1.overall_score - metrics2.overall_score),
        'category_comparison': {
            'fill_quality': {
                'puzzle1': metrics1.fill_score,
                'puzzle2': metrics2.fill_score,
                'winner': 'Puzzle 1' if metrics1.fill_score > metrics2.fill_score else 'Puzzle 2'
            },
            'construction': {
                'puzzle1': metrics1.construction_score,
                'puzzle2': metrics2.construction_score,
                'winner': 'Puzzle 1' if metrics1.construction_score > metrics2.construction_score else 'Puzzle 2'
            },
            'difficulty': {
                'puzzle1': metrics1.difficulty_score,
                'puzzle2': metrics2.difficulty_score,
                'more_difficult': 'Puzzle 1' if metrics1.difficulty_score > metrics2.difficulty_score else 'Puzzle 2'
            }
        },
        'strengths': {
            'puzzle1': [],
            'puzzle2': []
        },
        'weaknesses': {
            'puzzle1': [],
            'puzzle2': []
        }
    }
    
    # Identify strengths and weaknesses
    if metrics1.average_word_score > metrics2.average_word_score:
        comparison['strengths']['puzzle1'].append('Better word quality')
        comparison['weaknesses']['puzzle2'].append('Lower word quality')
    else:
        comparison['strengths']['puzzle2'].append('Better word quality')
        comparison['weaknesses']['puzzle1'].append('Lower word quality')
    
    if metrics1.three_letter_words < metrics2.three_letter_words:
        comparison['strengths']['puzzle1'].append('Fewer three-letter words')
        comparison['weaknesses']['puzzle2'].append('More three-letter words')
    else:
        comparison['strengths']['puzzle2'].append('Fewer three-letter words')
        comparison['weaknesses']['puzzle1'].append('More three-letter words')
    
    if metrics1.connectivity_score > metrics2.connectivity_score:
        comparison['strengths']['puzzle1'].append('Better grid connectivity')
        comparison['weaknesses']['puzzle2'].append('Weaker grid connectivity')
    else:
        comparison['strengths']['puzzle2'].append('Better grid connectivity')
        comparison['weaknesses']['puzzle1'].append('Weaker grid connectivity')
    
    return comparison
