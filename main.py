#!/usr/bin/env python3
"""
Crossword Puzzle Generator - Main Entry Point
A production-ready crossword and word-search puzzle generator.
"""

import sys
import os
import argparse
import yaml
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossword.grid import CrosswordGrid, GridSymmetry
from crossword.lexicon import Lexicon, WordlistLoader
from crossword.csp import CrosswordSolver, SolverConfig
from crossword.clue import ClueGenerator, ClueDifficulty
from crossword.export import ExportManager
from crossword.quality import QualityAnalyzer
from crossword.wordsearch import WordSearchGenerator


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure structured logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or return defaults."""
    default_config = {
        'grid': {
            'width': 15,
            'height': 15,
            'symmetry': 'rotational_180',
            'min_word_length': 3,
            'max_black_squares': 78
        },
        'solver': {
            'max_iterations': 100000,
            'time_limit_seconds': 300,
            'use_mrv_heuristic': True,
            'use_degree_heuristic': True,
            'use_forward_checking': True,
            'word_score_weight': 0.7,
            'crossing_weight': 0.3
        },
        'clues': {
            'difficulty': 'medium',
            'cryptic_mode': False,
            'allow_abbreviations': True,
            'allow_proper_nouns': True
        },
        'theme': {
            'entries': [],
            'enforce_symmetry': True
        },
        'wordlists': {
            'primary': 'data/wordlist.json',
            'domain_specific': [],
            'banned_words': []
        },
        'export': {
            'formats': ['json', 'pdf'],
            'output_dir': 'output',
            'include_construction_log': False
        },
        'random_seed': None,
        'logging': {
            'level': 'INFO',
            'file': None,
            'trace_file': None
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                user_config = yaml.safe_load(f)
            else:
                user_config = json.load(f)
        
        # Merge configurations (deep merge for nested dicts)
        def merge_config(default, user):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_config(default[key], value)
                else:
                    default[key] = value
        
        merge_config(default_config, user_config)
    
    return default_config


def create_crossword(config: Dict[str, Any]) -> bool:
    """Generate a crossword puzzle based on configuration."""
    logger = logging.getLogger(__name__)
    
    try:
        # Set random seed if specified
        if config['random_seed'] is not None:
            import random
            import numpy as np
            random.seed(config['random_seed'])
            np.random.seed(config['random_seed'])
        
        # Initialize grid
        symmetry = GridSymmetry.from_string(config['grid']['symmetry'])
        grid = CrosswordGrid(
            width=config['grid']['width'],
            height=config['grid']['height'],
            symmetry=symmetry,
            min_word_length=config['grid']['min_word_length']
        )
        
        # Load wordlists
        logger.info("Loading wordlists...")
        wordlist_loader = WordlistLoader()
        lexicon = wordlist_loader.load_primary_wordlist(config['wordlists']['primary'])
        
        # Load domain-specific wordlists
        for domain_path in config['wordlists']['domain_specific']:
            domain_lexicon = wordlist_loader.load_wordlist(domain_path)
            lexicon.merge(domain_lexicon)
        
        # Apply banned words
        if config['wordlists']['banned_words']:
            banned_words = set()
            for banned_path in config['wordlists']['banned_words']:
                if os.path.exists(banned_path):
                    with open(banned_path, 'r', encoding='utf-8') as f:
                        banned_words.update(word.strip().upper() for word in f)
            lexicon.remove_words(banned_words)
        
        # Configure solver
        solver_config = SolverConfig(
            max_iterations=config['solver']['max_iterations'],
            time_limit=config['solver']['time_limit_seconds'],
            use_mrv=config['solver']['use_mrv_heuristic'],
            use_degree=config['solver']['use_degree_heuristic'],
            use_forward_checking=config['solver']['use_forward_checking'],
            word_score_weight=config['solver']['word_score_weight'],
            crossing_weight=config['solver']['crossing_weight']
        )
        
        # Add theme entries if specified
        theme_entries = config.get('theme', {}).get('entries', [])
        if theme_entries:
            logger.info(f"Adding {len(theme_entries)} theme entries...")
            for theme_entry in theme_entries:
                grid.add_theme_entry(
                    word=theme_entry['word'].upper(),
                    row=theme_entry['row'],
                    col=theme_entry['col'],
                    direction=theme_entry['direction']
                )
        
        # Initialize solver
        solver = CrosswordSolver(grid, lexicon, solver_config)
        
        # Set up tracing if requested
        if config['logging'].get('trace_file'):
            solver.enable_tracing(config['logging']['trace_file'])
        
        # Solve the crossword
        logger.info("Starting crossword generation...")
        success = solver.solve()
        
        if not success:
            logger.error("Failed to generate a valid crossword puzzle")
            return False
        
        # Generate clues
        logger.info("Generating clues...")
        difficulty = ClueDifficulty.from_string(config['clues']['difficulty'])
        clue_generator = ClueGenerator(
            difficulty=difficulty,
            cryptic_mode=config['clues']['cryptic_mode'],
            allow_abbreviations=config['clues']['allow_abbreviations'],
            allow_proper_nouns=config['clues']['allow_proper_nouns']
        )
        
        clues = clue_generator.generate_clues(grid, lexicon)
        
        # Analyze quality
        logger.info("Analyzing puzzle quality...")
        quality_analyzer = QualityAnalyzer()
        quality_metrics = quality_analyzer.analyze(grid, lexicon, clues)
        
        logger.info(f"Puzzle quality score: {quality_metrics.overall_score:.2f}")
        logger.info(f"Fill quality: {quality_metrics.fill_score:.2f}")
        logger.info(f"Difficulty estimate: {quality_metrics.difficulty_score:.2f}")
        
        # Export puzzle
        logger.info("Exporting puzzle...")
        export_manager = ExportManager(config['export']['output_dir'])
        
        puzzle_data = {
            'grid': grid,
            'clues': clues,
            'quality_metrics': quality_metrics,
            'construction_log': solver.get_construction_log() if config['export']['include_construction_log'] else None
        }
        
        for export_format in config['export']['formats']:
            output_path = export_manager.export(puzzle_data, export_format)
            logger.info(f"Exported {export_format.upper()} to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating crossword: {e}", exc_info=True)
        return False


def create_wordsearch(config: Dict[str, Any]) -> bool:
    """Generate a word search puzzle based on configuration."""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize word search generator
        ws_config = config.get('wordsearch', {})
        generator = WordSearchGenerator(
            width=ws_config.get('width', 20),
            height=ws_config.get('height', 20),
            directions=ws_config.get('directions', ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
            allow_overlaps=ws_config.get('allow_overlaps', True),
            min_overlap=ws_config.get('min_overlap', 2)
        )
        
        # Load word list
        word_file = ws_config.get('word_file', 'data/wordsearch_words.txt')
        if not os.path.exists(word_file):
            logger.error(f"Word search word file not found: {word_file}")
            return False
        
        with open(word_file, 'r', encoding='utf-8') as f:
            words = [line.strip().upper() for line in f if line.strip()]
        
        # Generate puzzle
        logger.info(f"Generating word search with {len(words)} words...")
        puzzle = generator.generate(words)
        
        if not puzzle:
            logger.error("Failed to generate word search puzzle")
            return False
        
        # Export puzzle
        export_manager = ExportManager(config['export']['output_dir'])
        output_path = export_manager.export_wordsearch(puzzle)
        logger.info(f"Exported word search to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating word search: {e}", exc_info=True)
        return False


def main():
    """Main entry point for the crossword generator."""
    parser = argparse.ArgumentParser(
        description="Generate crossword puzzles and word searches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config.yaml
  python main.py --grid-size 13x13 --symmetry rotational_180
  python main.py --wordsearch --word-file words.txt
  python main.py --theme-entries "PYTHON,5,5,across" "CODING,8,2,down"
        """
    )
    
    parser.add_argument('--config', help='Configuration file (YAML or JSON)')
    parser.add_argument('--grid-size', help='Grid size (e.g., 15x15)')
    parser.add_argument('--symmetry', choices=['rotational_180', 'barred', 'asymmetric'],
                       help='Grid symmetry type')
    parser.add_argument('--min-word-length', type=int, help='Minimum word length')
    parser.add_argument('--wordlist', help='Primary wordlist file')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                       help='Clue difficulty level')
    parser.add_argument('--cryptic', action='store_true', help='Enable cryptic clues')
    parser.add_argument('--theme-entries', nargs='*', 
                       help='Theme entries (format: WORD,row,col,direction)')
    parser.add_argument('--max-iterations', type=int, help='Maximum solver iterations')
    parser.add_argument('--time-limit', type=int, help='Time limit in seconds')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--formats', nargs='*', 
                       choices=['json', 'puz', 'ipuz', 'pdf', 'png'],
                       help='Export formats')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible puzzles')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--wordsearch', action='store_true', 
                       help='Generate word search instead of crossword')
    parser.add_argument('--word-file', help='Word file for word search')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.grid_size:
        width, height = map(int, args.grid_size.split('x'))
        config['grid']['width'] = width
        config['grid']['height'] = height
    
    if args.symmetry:
        config['grid']['symmetry'] = args.symmetry
    
    if args.min_word_length:
        config['grid']['min_word_length'] = args.min_word_length
    
    if args.wordlist:
        config['wordlists']['primary'] = args.wordlist
    
    if args.difficulty:
        config['clues']['difficulty'] = args.difficulty
    
    if args.cryptic:
        config['clues']['cryptic_mode'] = True
    
    if args.theme_entries:
        theme_entries = []
        for entry in args.theme_entries:
            parts = entry.split(',')
            if len(parts) == 4:
                theme_entries.append({
                    'word': parts[0],
                    'row': int(parts[1]),
                    'col': int(parts[2]),
                    'direction': parts[3]
                })
        config['theme']['entries'] = theme_entries
    
    if args.max_iterations:
        config['solver']['max_iterations'] = args.max_iterations
    
    if args.time_limit:
        config['solver']['time_limit_seconds'] = args.time_limit
    
    if args.output_dir:
        config['export']['output_dir'] = args.output_dir
    
    if args.formats:
        config['export']['formats'] = args.formats
    
    if args.seed is not None:
        config['random_seed'] = args.seed
    
    config['logging']['level'] = args.log_level
    if args.log_file:
        config['logging']['file'] = args.log_file
    
    if args.wordsearch:
        if args.word_file:
            config.setdefault('wordsearch', {})['word_file'] = args.word_file
    
    # Setup logging
    setup_logging(config['logging']['level'], config['logging']['file'])
    
    # Create output directory
    os.makedirs(config['export']['output_dir'], exist_ok=True)
    
    # Generate puzzle
    if args.wordsearch:
        success = create_wordsearch(config)
    else:
        success = create_crossword(config)
    
    if success:
        print("Puzzle generation completed successfully!")
        return 0
    else:
        print("Puzzle generation failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
