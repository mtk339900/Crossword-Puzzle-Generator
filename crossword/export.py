"""
Export module for crossword puzzles.
Handles exporting to various formats: JSON, PUZ, IPUZ, PDF, PNG.
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import io
import base64

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch, mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from crossword.grid import CrosswordGrid, Direction
from crossword.clue import ClueEntry
from crossword.quality import QualityMetrics
from crossword.wordsearch import WordSearchPuzzle


class ExportManager:
    """Manages export of crossword puzzles to various formats."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize export manager.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check available libraries
        self.pdf_available = REPORTLAB_AVAILABLE
        self.image_available = PIL_AVAILABLE
        
        if not self.pdf_available:
            self.logger.warning("ReportLab not available - PDF export disabled")
        
        if not self.image_available:
            self.logger.warning("PIL not available - PNG export disabled")
    
    def export(self, puzzle_data: Dict[str, Any], format_type: str, 
              filename: Optional[str] = None) -> str:
        """Export puzzle to specified format.
        
        Args:
            puzzle_data: Dictionary containing grid, clues, quality_metrics, etc.
            format_type: Export format (json, puz, ipuz, pdf, png)
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to exported file
        """
        format_type = format_type.lower()
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crossword_{timestamp}.{format_type}"
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            if format_type == 'json':
                self._export_json(puzzle_data, output_path)
            elif format_type == 'puz':
                self._export_puz(puzzle_data, output_path)
            elif format_type == 'ipuz':
                self._export_ipuz(puzzle_data, output_path)
            elif format_type == 'pdf':
                if self.pdf_available:
                    self._export_pdf(puzzle_data, output_path)
                else:
                    raise RuntimeError("PDF export requires ReportLab library")
            elif format_type == 'png':
                if self.image_available:
                    self._export_png(puzzle_data, output_path)
                else:
                    raise RuntimeError("PNG export requires PIL library")
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            self.logger.info(f"Exported {format_type.upper()} to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting {format_type}: {e}")
            raise
    
    def _export_json(self, puzzle_data: Dict[str, Any], output_path: str):
        """Export puzzle as JSON."""
        grid = puzzle_data['grid']
        clues = puzzle_data['clues']
        quality_metrics = puzzle_data.get('quality_metrics')
        construction_log = puzzle_data.get('construction_log')
        
        # Build JSON structure
        export_data = {
            'metadata': {
                'generator': 'crossword-generator',
                'created': datetime.now().isoformat(),
                'version': '1.0'
            },
            'puzzle': {
                'grid': grid.to_dict(),
                'clues': {
                    'across': {},
                    'down': {}
                }
            }
        }
        
        # Add clues
        for slot, clue_entry in clues.items():
            clue_data = {
                'clue': clue_entry.clue,
                'answer': clue_entry.word,
                'type': clue_entry.clue_type.value,
                'difficulty': clue_entry.difficulty.value,
                'confidence': clue_entry.confidence
            }
            
            if clue_entry.enumeration:
                clue_data['enumeration'] = clue_entry.enumeration
            
            if slot.direction == Direction.ACROSS:
                export_data['puzzle']['clues']['across'][str(slot.number)] = clue_data
            else:
                export_data['puzzle']['clues']['down'][str(slot.number)] = clue_data
        
        # Add quality metrics if available
        if quality_metrics:
            export_data['quality'] = {
                'overall_score': quality_metrics.overall_score,
                'fill_score': quality_metrics.fill_score,
                'difficulty_score': quality_metrics.difficulty_score,
                'theme_score': quality_metrics.theme_score,
                'metrics': quality_metrics.to_dict()
            }
        
        # Add construction log if available
        if construction_log:
            export_data['construction_log'] = construction_log
        
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_puz(self, puzzle_data: Dict[str, Any], output_path: str):
        """Export puzzle as PUZ format (simplified implementation)."""
        # Note: This is a simplified PUZ format implementation
        # A full implementation would require the complete AcrossLite specification
        
        grid = puzzle_data['grid']
        clues = puzzle_data['clues']
        
        # Basic PUZ structure (simplified)
        puz_data = {
            'title': 'Generated Crossword',
            'author': 'Crossword Generator',
            'copyright': f'© {datetime.now().year}',
            'width': grid.width,
            'height': grid.height,
            'grid': [],
            'clues': []
        }
        
        # Convert grid
        for row in range(grid.height):
            grid_row = []
            for col in range(grid.width):
                if grid.is_black_square(row, col):
                    grid_row.append('.')
                else:
                    letter = grid.get_letter(row, col)
                    grid_row.append(letter if letter else '-')
            puz_data['grid'].append(''.join(grid_row))
        
        # Convert clues
        across_clues = []
        down_clues = []
        
        for slot, clue_entry in clues.items():
            clue_text = f"{slot.number}. {clue_entry.clue}"
            if slot.direction == Direction.ACROSS:
                across_clues.append(clue_text)
            else:
                down_clues.append(clue_text)
        
        puz_data['clues'] = across_clues + down_clues
        
        # Write simplified PUZ file (as JSON for this implementation)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(puz_data, f, indent=2)
    
    def _export_ipuz(self, puzzle_data: Dict[str, Any], output_path: str):
        """Export puzzle as IPUZ format."""
        grid = puzzle_data['grid']
        clues = puzzle_data['clues']
        quality_metrics = puzzle_data.get('quality_metrics')
        
        # Build IPUZ structure
        ipuz_data = {
            'version': 'http://www.ipuz.org/v2',
            'kind': ['http://www.ipuz.org/crossword#1'],
            'title': 'Generated Crossword',
            'author': 'Crossword Generator',
            'copyright': f'© {datetime.now().year}',
            'publisher': 'Crossword Generator',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'dimensions': {
                'width': grid.width,
                'height': grid.height
            },
            'puzzle': [],
            'solution': [],
            'clues': {
                'Across': [],
                'Down': []
            }
        }
        
        # Add quality rating if available
        if quality_metrics:
            ipuz_data['difficulty'] = quality_metrics.difficulty_score
        
        # Build puzzle and solution grids
        for row in range(grid.height):
            puzzle_row = []
            solution_row = []
            
            for col in range(grid.width):
                if grid.is_black_square(row, col):
                    puzzle_row.append(0)  # Black square
                    solution_row.append(0)
                else:
                    number = grid.numbers[row][col]
                    letter = grid.get_letter(row, col)
                    
                    if number:
                        puzzle_row.append(number)
                    else:
                        puzzle_row.append(1)  # Empty numbered square
                    
                    solution_row.append(letter if letter else '')
            
            ipuz_data['puzzle'].append(puzzle_row)
            ipuz_data['solution'].append(solution_row)
        
        # Add clues
        for slot, clue_entry in clues.items():
            clue_item = [slot.number, clue_entry.clue]
            
            if slot.direction == Direction.ACROSS:
                ipuz_data['clues']['Across'].append(clue_item)
            else:
                ipuz_data['clues']['Down'].append(clue_item)
        
        # Sort clues by number
        ipuz_data['clues']['Across'].sort(key=lambda x: x[0])
        ipuz_data['clues']['Down'].sort(key=lambda x: x[0])
        
        # Write IPUZ file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ipuz_data, f, indent=2, ensure_ascii=False)
    
    def _export_pdf(self, puzzle_data: Dict[str, Any], output_path: str):
        """Export puzzle as PDF."""
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("ReportLab library required for PDF export")
        
        grid = puzzle_data['grid']
        clues = puzzle_data['clues']
        quality_metrics = puzzle_data.get('quality_metrics')
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        story.append(Paragraph("Crossword Puzzle", title_style))
        
        # Add quality metrics if available
        if quality_metrics:
            quality_text = f"Difficulty: {quality_metrics.difficulty_score:.1f}/10 | " \
                          f"Quality: {quality_metrics.overall_score:.1f}/100"
            story.append(Paragraph(quality_text, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Create grid table
        grid_data = self._create_pdf_grid_data(grid)
        grid_table = Table(grid_data, colWidths=[25] * grid.width, rowHeights=[25] * grid.height)
        
        # Style the grid
        grid_style = [
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
        
        # Add black squares
        for row in range(grid.height):
            for col in range(grid.width):
                if grid.is_black_square(row, col):
                    grid_style.append(('BACKGROUND', (col, row), (col, row), colors.black))
        
        grid_table.setStyle(TableStyle(grid_style))
        story.append(grid_table)
        story.append(Spacer(1, 30))
        
        # Add clues
        story.extend(self._create_pdf_clues(clues, styles))
        
        # Build PDF
        doc.build(story)
    
    def _create_pdf_grid_data(self, grid: CrosswordGrid) -> List[List[str]]:
        """Create grid data for PDF table."""
        grid_data = []
        
        for row in range(grid.height):
            row_data = []
            for col in range(grid.width):
                if grid.is_black_square(row, col):
                    row_data.append('')
                else:
                    number = grid.numbers[row][col]
                    if number:
                        row_data.append(str(number))
                    else:
                        row_data.append('')
            grid_data.append(row_data)
        
        return grid_data
    
    def _create_pdf_clues(self, clues: Dict, styles) -> List:
        """Create clue sections for PDF."""
        story_elements = []
        
        # Separate and sort clues
        across_clues = []
        down_clues = []
        
        for slot, clue_entry in clues.items():
            clue_text = f"{slot.number}. {clue_entry.clue}"
            if slot.direction == Direction.ACROSS:
                across_clues.append((slot.number, clue_text))
            else:
                down_clues.append((slot.number, clue_text))
        
        across_clues.sort(key=lambda x: x[0])
        down_clues.sort(key=lambda x: x[0])
        
        # Across clues
        story_elements.append(Paragraph("ACROSS", styles['Heading2']))
        for _, clue_text in across_clues:
            story_elements.append(Paragraph(clue_text, styles['Normal']))
        
        story_elements.append(Spacer(1, 20))
        
        # Down clues  
        story_elements.append(Paragraph("DOWN", styles['Heading2']))
        for _, clue_text in down_clues:
            story_elements.append(Paragraph(clue_text, styles['Normal']))
        
        return story_elements
    
    def _export_png(self, puzzle_data: Dict[str, Any], output_path: str):
        """Export puzzle as PNG image."""
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL library required for PNG export")
        
        grid = puzzle_data['grid']
        clues = puzzle_data['clues']
        
        # Image dimensions
        cell_size = 30
        number_font_size = 8
        grid_width = grid.width * cell_size
        grid_height = grid.height * cell_size
        
        # Create image with extra space for clues
        img_width = max(grid_width, 400)
        img_height = grid_height + 600  # Extra space for clues
        
        image = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Try to load fonts
        try:
            number_font = ImageFont.truetype('arial.ttf', number_font_size)
            clue_font = ImageFont.truetype('arial.ttf', 10)
            title_font = ImageFont.truetype('arial.ttf', 14)
        except:
            number_font = ImageFont.load_default()
            clue_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Draw grid
        for row in range(grid.height):
            for col in range(grid.width):
                x1 = col * cell_size
                y1 = row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                if grid.is_black_square(row, col):
                    # Black square
                    draw.rectangle([x1, y1, x2, y2], fill='black', outline='black')
                else:
                    # White square
                    draw.rectangle([x1, y1, x2, y2], fill='white', outline='black')
                    
                    # Add number if present
                    number = grid.numbers[row][col]
                    if number:
                        draw.text((x1 + 2, y1 + 2), str(number), fill='black', font=number_font)
        
        # Draw title
        title_y = grid_height + 20
        draw.text((10, title_y), "Crossword Puzzle", fill='black', font=title_font)
        
        # Draw clues
        clue_y = title_y + 40
        
        # Separate and sort clues
        across_clues = []
        down_clues = []
        
        for slot, clue_entry in clues.items():
            clue_text = f"{slot.number}. {clue_entry.clue}"
            if slot.direction == Direction.ACROSS:
                across_clues.append((slot.number, clue_text))
            else:
                down_clues.append((slot.number, clue_text))
        
        across_clues.sort(key=lambda x: x[0])
        down_clues.sort(key=lambda x: x[0])
        
        # Draw across clues
        draw.text((10, clue_y), "ACROSS", fill='black', font=title_font)
        clue_y += 25
        
        for _, clue_text in across_clues[:10]:  # Limit for space
            # Wrap long clues
            if len(clue_text) > 50:
                clue_text = clue_text[:47] + "..."
            draw.text((10, clue_y), clue_text, fill='black', font=clue_font)
            clue_y += 15
        
        clue_y += 10
        
        # Draw down clues
        draw.text((10, clue_y), "DOWN", fill='black', font=title_font)
        clue_y += 25
        
        for _, clue_text in down_clues[:10]:  # Limit for space
            # Wrap long clues
            if len(clue_text) > 50:
                clue_text = clue_text[:47] + "..."
            draw.text((10, clue_y), clue_text, fill='black', font=clue_font)
            clue_y += 15
        
        # Save image
        image.save(output_path, 'PNG')
    
    def export_wordsearch(self, puzzle: WordSearchPuzzle, filename: Optional[str] = None) -> str:
        """Export word search puzzle as JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wordsearch_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Create export data
        export_data = {
            'metadata': {
                'type': 'wordsearch',
                'generator': 'crossword-generator',
                'created': datetime.now().isoformat(),
                'version': '1.0'
            },
            'puzzle': puzzle.to_dict(),
            'answer_key': {
                'placed_words': [
                    {
                        'word': pw.word,
                        'start': [pw.start_row, pw.start_col],
                        'end': [pw.end_row, pw.end_col],
                        'direction': pw.direction.value
                    }
                    for pw in puzzle.placed_words
                ]
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported word search to {output_path}")
        return output_path
    
    def export_solution(self, puzzle_data: Dict[str, Any], format_type: str = 'json',
                       filename: Optional[str] = None) -> str:
        """Export puzzle solution separately."""
        grid = puzzle_data['grid']
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solution_{timestamp}.{format_type}"
        
        output_path = os.path.join(self.output_dir, filename)
        
        if format_type.lower() == 'json':
            solution_data = {
                'metadata': {
                    'type': 'solution',
                    'generator': 'crossword-generator',
                    'created': datetime.now().isoformat()
                },
                'solution': {
                    'grid': [],
                    'words': {}
                }
            }
            
            # Add solution grid
            for row in range(grid.height):
                grid_row = []
                for col in range(grid.width):
                    if grid.is_black_square(row, col):
                        grid_row.append(None)
                    else:
                        letter = grid.get_letter(row, col)
                        grid_row.append(letter)
                solution_data['solution']['grid'].append(grid_row)
            
            # Add word positions
            for slot in grid.get_filled_slots():
                solution_data['solution']['words'][slot.word] = {
                    'start_row': slot.start_row,
                    'start_col': slot.start_col,
                    'direction': slot.direction.value,
                    'length': slot.length,
                    'number': slot.number
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(solution_data, f, indent=2, ensure_ascii=False)
        
        elif format_type.lower() == 'png' and self.image_available:
            # Create solution image similar to puzzle but with letters filled in
            cell_size = 30
            img_width = grid.width * cell_size
            img_height = grid.height * cell_size
            
            image = Image.new('RGB', (img_width, img_height), 'white')
            draw = ImageDraw.Draw(image)
            
            try:
                letter_font = ImageFont.truetype('arial.ttf', 14)
            except:
                letter_font = ImageFont.load_default()
            
            # Draw solution grid with letters
            for row in range(grid.height):
                for col in range(grid.width):
                    x1 = col * cell_size
                    y1 = row * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size
                    
                    if grid.is_black_square(row, col):
                        draw.rectangle([x1, y1, x2, y2], fill='black', outline='black')
                    else:
                        draw.rectangle([x1, y1, x2, y2], fill='white', outline='black')
                        letter = grid.get_letter(row, col)
                        if letter:
                            # Center the letter in the cell
                            text_width = draw.textlength(letter, font=letter_font)
                            text_x = x1 + (cell_size - text_width) // 2
                            text_y = y1 + (cell_size - 16) // 2
                            draw.text((text_x, text_y), letter, fill='black', font=letter_font)
            
            image.save(output_path, 'PNG')
        
        else:
            raise ValueError(f"Unsupported solution format: {format_type}")
        
        self.logger.info(f"Exported solution to {output_path}")
        return output_path
    
    def export_construction_report(self, puzzle_data: Dict[str, Any], 
                                 filename: Optional[str] = None) -> str:
        """Export detailed construction report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"construction_report_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        grid = puzzle_data['grid']
        clues = puzzle_data['clues']
        quality_metrics = puzzle_data.get('quality_metrics')
        construction_log = puzzle_data.get('construction_log', [])
        
        report_data = {
            'metadata': {
                'type': 'construction_report',
                'generator': 'crossword-generator', 
                'created': datetime.now().isoformat(),
                'version': '1.0'
            },
            'puzzle_info': {
                'dimensions': f"{grid.width}x{grid.height}",
                'symmetry': grid.symmetry.value,
                'word_count': len(grid.get_all_slots()),
                'filled_count': len(grid.get_filled_slots()),
                'theme_entries': len(grid.theme_entries),
                'black_square_count': grid.get_black_square_count(),
                'white_square_count': grid.get_white_square_count()
            },
            'quality_analysis': quality_metrics.to_dict() if quality_metrics else {},
            'word_analysis': self._analyze_words(grid, clues),
            'construction_log': construction_log,
            'validation_results': grid.validate_grid()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported construction report to {output_path}")
        return output_path
    
    def _analyze_words(self, grid: CrosswordGrid, clues: Dict) -> Dict[str, Any]:
        """Analyze words in the puzzle for the construction report."""
        filled_slots = grid.get_filled_slots()
        
        analysis = {
            'word_lengths': {},
            'letter_frequency': {},
            'crossing_analysis': {},
            'clue_analysis': {
                'by_type': {},
                'by_difficulty': {},
                'average_confidence': 0.0
            }
        }
        
        # Word length analysis
        for slot in filled_slots:
            length = slot.length
            analysis['word_lengths'][length] = analysis['word_lengths'].get(length, 0) + 1
        
        # Letter frequency analysis
        for slot in filled_slots:
            for letter in slot.word:
                analysis['letter_frequency'][letter] = analysis['letter_frequency'].get(letter, 0) + 1
        
        # Crossing analysis
        total_crossings = 0
        crossing_counts = {}
        
        for slot in filled_slots:
            crossings = len(grid.get_crossing_slots(slot))
            total_crossings += crossings
            crossing_counts[crossings] = crossing_counts.get(crossings, 0) + 1
        
        analysis['crossing_analysis'] = {
            'average_crossings': total_crossings / len(filled_slots) if filled_slots else 0,
            'distribution': crossing_counts
        }
        
        # Clue analysis
        if clues:
            confidence_sum = 0
            for clue_entry in clues.values():
                clue_type = clue_entry.clue_type.value
                difficulty = clue_entry.difficulty.value
                
                analysis['clue_analysis']['by_type'][clue_type] = \
                    analysis['clue_analysis']['by_type'].get(clue_type, 0) + 1
                analysis['clue_analysis']['by_difficulty'][difficulty] = \
                    analysis['clue_analysis']['by_difficulty'].get(difficulty, 0) + 1
                
                confidence_sum += clue_entry.confidence
            
            analysis['clue_analysis']['average_confidence'] = confidence_sum / len(clues)
        
        return analysis
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        formats = ['json', 'puz', 'ipuz']
        
        if self.pdf_available:
            formats.append('pdf')
        
        if self.image_available:
            formats.append('png')
        
        return formats
    
    def validate_export_data(self, puzzle_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate puzzle data before export."""
        issues = []
        
        # Check required fields
        if 'grid' not in puzzle_data:
            issues.append("Missing grid data")
        
        if 'clues' not in puzzle_data:
            issues.append("Missing clues data")
        
        # Validate grid
        grid = puzzle_data.get('grid')
        if grid:
            grid_issues = grid.validate_grid()
            issues.extend(grid_issues)
        
        # Validate clues
        clues = puzzle_data.get('clues', {})
        if grid and clues:
            filled_slots = grid.get_filled_slots()
            
            # Check that all filled slots have clues
            for slot in filled_slots:
                if slot not in clues:
                    issues.append(f"Missing clue for slot at ({slot.start_row},{slot.start_col}) {slot.direction.value}")
            
            # Check that all clues correspond to filled slots
            for slot in clues:
                if slot not in filled_slots:
                    issues.append(f"Clue exists for unfilled slot at ({slot.start_row},{slot.start_col}) {slot.direction.value}")
        
        return len(issues) == 0, issues
