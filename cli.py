"""
Command-line interface module for the crossword generator.
Provides additional CLI utilities and commands.
"""

import sys
import argparse
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from crossword.grid import CrosswordGrid, GridSymmetry, initialize_grid
from crossword.lexicon import WordlistLoader, create_sample_wordlist
from crossword.csp import solve_crossword, SolverConfig
from crossword.clue import ClueGenerator, ClueDifficulty, create_sample_clue_database
from crossword.export import ExportManager
from crossword.quality import QualityAnalyzer, analyze_puzzle_difficulty
from crossword.wordsearch import WordSearchGenerator, create_themed_word_search


class CrosswordCLI:
    """Command-line interface for crossword operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_sample_files(self, output_dir: str = "data"):
        """Create sample configuration and data files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create sample configuration
        sample_config = {
            'grid': {
                'width': 15,
                'height': 15,
                'symmetry': 'rotational_180',
                'min_word_length': 3,
                'max_black_squares': 78
            },
            'solver': {
                'max_iterations': 50000,
                'time_limit_seconds': 180,
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
                'entries': [
                    {
                        'word': 'PYTHON',
                        'row': 7,
                        'col': 4,
                        'direction': 'across'
                    }
                ],
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
                'include_construction_log': True
            },
            'logging': {
                'level': 'INFO',
                'file': None,
                'trace_file': None
            }
        }
        
        config_path = os.path.join(output_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False)
        
        print(f"Created sample configuration: {config_path}")
        
        # Create sample wordlist
        lexicon = create_sample_wordlist()
        wordlist_path = os.path.join(output_dir, "wordlist.json")
        
        loader = WordlistLoader()
        loader.save_wordlist(lexicon, wordlist_path, 'json')
        
        print(f"Created sample wordlist: {wordlist_path}")
        
        # Create sample clue database
        clue_db = create_sample_clue_database()
        clue_db_path = os.path.join(output_dir, "clues.json")
        clue_db.save_to_file(clue_db_path)
        
        print(f"Created sample clue database: {clue_db_path}")
        
        # Create sample word search word list
        wordsearch_words = [
            "COMPUTER", "KEYBOARD", "MONITOR", "PRINTER", "SCANNER",
            "INTERNET", "WEBSITE", "EMAIL", "PASSWORD", "SOFTWARE",
            "HARDWARE", "NETWORK", "DATABASE", "PROGRAM", "CODING"
        ]
        
        ws_path = os.path.join(output_dir, "wordsearch_words.txt")
        with open(ws_path, 'w') as f:
            for word in wordsearch_words:
                f.write(f"{word}\n")
        
        print(f"Created sample word search list: {ws_path}")
        
        print("\nSample files created successfully!")
        print(f"You can now run: python main.py --config {config_path}")
    
    def validate_wordlist(self, wordlist_path: str) -> bool:
        """Validate a wordlist file."""
        try:
            loader = WordlistLoader()
            lexicon = loader.load_wordlist(wordlist_path)
            
            stats = lexicon.get_statistics()
            
            print(f"Wordlist Validation Report: {wordlist_path}")
            print(f"Total words: {stats['total_words']}")
            print(f"Word length distribution: {stats['by_length']}")
            print(f"Score range: {stats['score_range'][0]:.1f} - {stats['score_range'][1]:.1f}")
            print(f"Proper nouns: {stats['proper_nouns']}")
            print(f"Abbreviations: {stats['abbreviations']}")
            print(f"Difficulty distribution: {dict(stats['by_difficulty'])}")
            
            # Check for potential issues
            issues = []
            
            if stats['total_words'] < 1000:
                issues.append("Low word count - may limit puzzle generation")
            
            length_3_count = stats['by_length'].get(3, 0)
            if length_3_count < stats['total_words'] * 0.1:
                issues.append("Few 3-letter words - may cause filling problems")
            
            if stats['score_range'][1] - stats['score_range'][0] < 50:
                issues.append("Narrow score range - limited word quality variation")
            
            if issues:
                print("\nPotential issues:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("\nWordlist appears to be well-formed.")
            
            return len(issues) == 0
            
        except Exception as e:
            print(f"Error validating wordlist: {e}")
            return False
    
    def analyze_puzzle(self, puzzle_file: str, wordlist_file: str = None) -> bool:
        """Analyze an existing puzzle."""
        try:
            # Load puzzle
            with open(puzzle_file, 'r') as f:
                puzzle_data = json.load(f)
            
            # Reconstruct grid (simplified for this example)
            if 'puzzle' not in puzzle_data or 'grid' not in puzzle_data['puzzle']:
                print("Invalid puzzle file format")
                return False
            
            grid_data = puzzle_data['puzzle']['grid']
            grid = CrosswordGrid(
                width=grid_data['width'],
                height=grid_data['height'],
                symmetry=GridSymmetry.from_string(grid_data['symmetry'])
            )
            
            # Load wordlist for analysis
            if wordlist_file:
                loader = WordlistLoader()
                lexicon = loader.load_wordlist(wordlist_file)
            else:
                lexicon = create_sample_wordlist()
            
            # Perform analysis
            analyzer = QualityAnalyzer()
            metrics = analyzer.analyze(grid, lexicon)
            
            # Generate report
            report = analyzer.generate_quality_report(metrics)
            print(report)
            
            # Save detailed analysis
            analysis_file = puzzle_file.replace('.json', '_analysis.json')
            with open(analysis_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            
            print(f"\nDetailed analysis saved to: {analysis_file}")
            
            return True
            
        except Exception as e:
            print(f"Error analyzing puzzle: {e}")
            return False
    
    def convert_wordlist(self, input_file: str, output_file: str, 
                        input_format: str = None, output_format: str = None) -> bool:
        """Convert wordlist between formats."""
        try:
            # Auto-detect formats if not specified
            if not input_format:
                if input_file.endswith('.json'):
                    input_format = 'json'
                elif input_file.endswith('.csv'):
                    input_format = 'csv'
                elif input_file.endswith('.tsv'):
                    input_format = 'tsv'
                else:
                    input_format = 'text'
            
            if not output_format:
                if output_file.endswith('.json'):
                    output_format = 'json'
                elif output_file.endswith('.csv'):
                    output_format = 'csv'
                elif output_file.endswith('.tsv'):
                    output_format = 'tsv'
                else:
                    output_format = 'text'
            
            # Load and save
            loader = WordlistLoader()
            lexicon = loader.load_wordlist(input_file)
            loader.save_wordlist(lexicon, output_file, output_format)
            
            print(f"Converted {input_file} ({input_format}) -> {output_file} ({output_format})")
            print(f"Processed {len(lexicon)} words")
            
            return True
            
        except Exception as e:
            print(f"Error converting wordlist: {e}")
            return False
    
    def benchmark_solver(self, config_file: str = None, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark solver performance."""
        print(f"Running solver benchmark ({iterations} iterations)...")
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
        else:
            # Use default config
            config = {
                'grid': {'width': 15, 'height': 15, 'symmetry': 'rotational_180'},
                'wordlists': {'primary': None},
                'solver': {'max_iterations': 10000, 'time_limit_seconds': 60}
            }
        
        # Create sample lexicon if no wordlist specified
        if not config['wordlists']['primary'] or not os.path.exists(config['wordlists']['primary']):
            lexicon = create_sample_wordlist()
        else:
            loader = WordlistLoader()
            lexicon = loader.load_wordlist(config['wordlists']['primary'])
        
        results = {
            'successful_runs': 0,
            'failed_runs': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'average_iterations': 0.0,
            'average_backtracks': 0.0,
            'average_fill_percentage': 0.0,
            'best_quality': 0.0,
            'worst_quality': 100.0
        }
        
        for i in range(iterations):
            print(f"  Run {i+1}/{iterations}...", end=' ')
            
            try:
                # Initialize grid
                grid = initialize_grid(
                    config['grid']['width'],
                    config['grid']['height'],
                    GridSymmetry.from_string(config['grid']['symmetry'])
                )
                
                # Solve
                solver_config = SolverConfig(
                    max_iterations=config['solver']['max_iterations'],
                    time_limit=config['solver']['time_limit_seconds']
                )
                
                success, solver = solve_crossword(grid, lexicon, 'basic', **solver_config.__dict__)
                stats = solver.get_solver_statistics()
                
                if success:
                    results['successful_runs'] += 1
                    results['total_time'] += stats['elapsed_time']
                    results['average_iterations'] += stats['iterations']
                    results['average_backtracks'] += stats['backtracks']
                    results['average_fill_percentage'] += stats['fill_percentage']
                    
                    # Quick quality assessment
                    analyzer = QualityAnalyzer()
                    metrics = analyzer.analyze(grid, lexicon)
                    results['best_quality'] = max(results['best_quality'], metrics.overall_score)
                    results['worst_quality'] = min(results['worst_quality'], metrics.overall_score)
                    
                    print(f"SUCCESS ({stats['elapsed_time']:.1f}s, {stats['fill_percentage']:.1f}% fill)")
                else:
                    results['failed_runs'] += 1
                    print("FAILED")
                
            except Exception as e:
                results['failed_runs'] += 1
                print(f"ERROR: {e}")
        
        # Calculate averages
        if results['successful_runs'] > 0:
            results['average_time'] = results['total_time'] / results['successful_runs']
            results['average_iterations'] /= results['successful_runs']
            results['average_backtracks'] /= results['successful_runs']
            results['average_fill_percentage'] /= results['successful_runs']
        
        # Print summary
        print(f"\nBenchmark Results:")
        print(f"  Success Rate: {results['successful_runs']}/{iterations} ({results['successful_runs']/iterations*100:.1f}%)")
        print(f"  Average Time: {results['average_time']:.2f}s")
        print(f"  Average Iterations: {results['average_iterations']:.0f}")
        print(f"  Average Backtracks: {results['average_backtracks']:.0f}")
        print(f"  Average Fill: {results['average_fill_percentage']:.1f}%")
        
        if results['successful_runs'] > 0:
            print(f"  Quality Range: {results['worst_quality']:.1f} - {results['best_quality']:.1f}")
        
        return results
    
    def create_themed_puzzle(self, theme: str, output_dir: str = "output") -> bool:
        """Create a themed crossword puzzle."""
        try:
            print(f"Creating themed puzzle: {theme}")
            
            # This is a simplified themed puzzle creator
            # A full implementation would use comprehensive theme databases
            theme_words = {
                'programming': [
                    ('PYTHON', 'Programming language'),
                    ('CODE', 'Program instructions'),
                    ('DEBUG', 'Fix program errors'),
                    ('ARRAY', 'Data structure'),
                    ('LOOP', 'Repetitive structure'),
                    ('CLASS', 'Object template'),
                    ('METHOD', 'Object function'),
                    ('STRING', 'Text data type')
                ],
                'science': [
                    ('ATOM', 'Basic matter unit'),
                    ('CELL', 'Life unit'),
                    ('DNA', 'Genetic code'),
                    ('ORBIT', 'Planetary path'),
                    ('LASER', 'Focused light beam'),
                    ('ENERGY', 'Capacity for work'),
                    ('MATTER', 'Physical substance'),
                    ('PHYSICS', 'Natural science')
                ],
                'music': [
                    ('PIANO', 'Keyboard instrument'),
                    ('GUITAR', 'String instrument'),
                    ('VIOLIN', 'Bowed string instrument'),
                    ('RHYTHM', 'Musical timing'),
                    ('MELODY', 'Musical tune'),
                    ('HARMONY', 'Musical combination'),
                    ('TEMPO', 'Musical speed'),
                    ('CHORD', 'Note combination')
                ]
            }
            
            if theme.lower() not in theme_words:
                available = ', '.join(theme_words.keys())
                print(f"Theme '{theme}' not available. Available themes: {available}")
                return False
            
            # Create configuration with theme entries
            theme_entries = []
            words, definitions = zip(*theme_words[theme.lower()][:4])  # Use first 4 as themes
            
            # Place theme entries symmetrically
            theme_entries = [
                {'word': words[0], 'row': 3, 'col': 4, 'direction': 'across'},
                {'word': words[1], 'row': 7, 'col': 2, 'direction': 'down'},
                {'word': words[2], 'row': 11, 'col': 3, 'direction': 'across'},
                {'word': words[3], 'row': 5, 'col': 10, 'direction': 'down'}
            ]
            
            config = {
                'grid': {
                    'width': 15,
                    'height': 15,
                    'symmetry': 'rotational_180',
                    'min_word_length': 3
                },
                'theme': {
                    'entries': theme_entries,
                    'enforce_symmetry': True
                },
                'wordlists': {'primary': None},
                'export': {
                    'formats': ['json', 'pdf'],
                    'output_dir': output_dir
                }
            }
            
            # Use the main creation function
            from main import create_crossword
            success = create_crossword(config)
            
            if success:
                print(f"Themed puzzle created successfully in {output_dir}")
            else:
                print("Failed to create themed puzzle")
            
            return success
            
        except Exception as e:
            print(f"Error creating themed puzzle: {e}")
            return False
    
    def interactive_mode(self):
        """Run interactive puzzle creation mode."""
        print("Interactive Crossword Generator")
        print("=" * 30)
        print()
        
        # Get basic parameters
        try:
            width = int(input("Grid width (default 15): ") or "15")
            height = int(input("Grid height (default 15): ") or "15")
            
            print("\nSymmetry options:")
            print("1. Rotational 180°")
            print("2. Barred")
            print("3. Asymmetric")
            symmetry_choice = input("Choose symmetry (1-3, default 1): ") or "1"
            
            symmetry_map = {
                "1": "rotational_180",
                "2": "barred", 
                "3": "asymmetric"
            }
            symmetry = symmetry_map.get(symmetry_choice, "rotational_180")
            
            print("\nDifficulty options:")
            print("1. Easy")
            print("2. Medium") 
            print("3. Hard")
            difficulty_choice = input("Choose difficulty (1-3, default 2): ") or "2"
            
            difficulty_map = {
                "1": "easy",
                "2": "medium",
                "3": "hard"
            }
            difficulty = difficulty_map.get(difficulty_choice, "medium")
            
            # Theme entries
            theme_entries = []
            add_themes = input("\nAdd theme entries? (y/n, default n): ").lower() == 'y'
            
            if add_themes:
                print("Enter theme entries (format: WORD,row,col,direction)")
                print("Enter empty line to finish")
                
                while True:
                    entry = input("Theme entry: ").strip()
                    if not entry:
                        break
                    
                    try:
                        parts = entry.split(',')
                        if len(parts) == 4:
                            theme_entries.append({
                                'word': parts[0].strip().upper(),
                                'row': int(parts[1].strip()),
                                'col': int(parts[2].strip()),
                                'direction': parts[3].strip().lower()
                            })
                            print(f"  Added: {parts[0]} at ({parts[1]},{parts[2]}) {parts[3]}")
                        else:
                            print("  Invalid format, skipped")
                    except ValueError:
                        print("  Invalid entry, skipped")
            
            # Export formats
            print("\nExport formats:")
            print("1. JSON only")
            print("2. JSON + PDF")
            print("3. All formats")
            format_choice = input("Choose formats (1-3, default 2): ") or "2"
            
            format_map = {
                "1": ["json"],
                "2": ["json", "pdf"],
                "3": ["json", "pdf", "png", "ipuz"]
            }
            formats = format_map.get(format_choice, ["json", "pdf"])
            
            # Output directory
            output_dir = input("\nOutput directory (default 'output'): ") or "output"
            
            # Build configuration
            config = {
                'grid': {
                    'width': width,
                    'height': height,
                    'symmetry': symmetry,
                    'min_word_length': 3
                },
                'solver': {
                    'max_iterations': 50000,
                    'time_limit_seconds': 300
                },
                'clues': {
                    'difficulty': difficulty,
                    'cryptic_mode': False,
                    'allow_abbreviations': True,
                    'allow_proper_nouns': True
                },
                'theme': {
                    'entries': theme_entries,
                    'enforce_symmetry': True
                },
                'wordlists': {
                    'primary': None  # Will use fallback
                },
                'export': {
                    'formats': formats,
                    'output_dir': output_dir
                }
            }
            
            print(f"\nGenerating {width}x{height} {symmetry} puzzle...")
            print(f"Difficulty: {difficulty}")
            print(f"Theme entries: {len(theme_entries)}")
            print(f"Export formats: {', '.join(formats)}")
            print()
            
            # Generate puzzle
            from main import create_crossword
            success = create_crossword(config)
            
            if success:
                print("\nPuzzle generated successfully!")
                print(f"Files saved to: {output_dir}")
            else:
                print("\nFailed to generate puzzle.")
                print("Try reducing theme entries or relaxing constraints.")
            
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
        except Exception as e:
            print(f"\nError in interactive mode: {e}")


def main():
    """CLI entry point for utility functions."""
    parser = argparse.ArgumentParser(
        description="Crossword Generator CLI Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  sample-files    Create sample configuration and data files
  validate        Validate a wordlist file
  analyze         Analyze an existing puzzle
  convert         Convert wordlist between formats
  benchmark       Benchmark solver performance
  themed          Create a themed puzzle
  interactive     Interactive puzzle creation mode

Examples:
  python cli.py sample-files
  python cli.py validate data/wordlist.json
  python cli.py analyze puzzle.json --wordlist data/wordlist.json
  python cli.py convert input.txt output.json
  python cli.py benchmark --config config.yaml --iterations 5
  python cli.py themed programming
  python cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Sample files command
    sample_parser = subparsers.add_parser('sample-files', help='Create sample files')
    sample_parser.add_argument('--output-dir', default='data', help='Output directory')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate wordlist')
    validate_parser.add_argument('wordlist', help='Wordlist file to validate')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze puzzle')
    analyze_parser.add_argument('puzzle', help='Puzzle file to analyze')
    analyze_parser.add_argument('--wordlist', help='Wordlist for analysis')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert wordlist format')
    convert_parser.add_argument('input', help='Input wordlist file')
    convert_parser.add_argument('output', help='Output wordlist file')
    convert_parser.add_argument('--input-format', help='Input format (json/csv/tsv/text)')
    convert_parser.add_argument('--output-format', help='Output format (json/csv/tsv/text)')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark solver')
    benchmark_parser.add_argument('--config', help='Configuration file')
    benchmark_parser.add_argument('--iterations', type=int, default=10, help='Number of test runs')
    
    # Themed command
    themed_parser = subparsers.add_parser('themed', help='Create themed puzzle')
    themed_parser.add_argument('theme', help='Theme name')
    themed_parser.add_argument('--output-dir', default='output', help='Output directory')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    cli = CrosswordCLI()
    
    try:
        if args.command == 'sample-files':
            cli.create_sample_files(args.output_dir)
        
        elif args.command == 'validate':
            success = cli.validate_wordlist(args.wordlist)
            return 0 if success else 1
        
        elif args.command == 'analyze':
            success = cli.analyze_puzzle(args.puzzle, args.wordlist)
            return 0 if success else 1
        
        elif args.command == 'convert':
            success = cli.convert_wordlist(
                args.input, args.output, 
                args.input_format, args.output_format
            )
            return 0 if success else 1
        
        elif args.command == 'benchmark':
            cli.benchmark_solver(args.config, args.iterations)
        
        elif args.command == 'themed':
            success = cli.create_themed_puzzle(args.theme, args.output_dir)
            return 0 if success else 1
        
        elif args.command == 'interactive':
            cli.interactive_mode()
        
        else:
            print(f"Unknown command: {args.command}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
