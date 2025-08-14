"""
Constraint Satisfaction Problem (CSP) solver for crossword generation.
Implements backtracking with heuristics and constraint propagation.
"""

import time
import random
import logging
from typing import List, Dict, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from crossword.grid import CrosswordGrid, GridSlot, Direction
from crossword.lexicon import Lexicon


class HeuristicType(Enum):
    """Types of heuristics for variable ordering."""
    MRV = "most_remaining_values"       # Most Remaining Values
    DEGREE = "degree"                   # Degree heuristic
    MRV_DEGREE = "mrv_degree"          # Combined MRV + Degree
    RANDOM = "random"                   # Random ordering


@dataclass
class SolverConfig:
    """Configuration for the crossword solver."""
    max_iterations: int = 100000
    time_limit: float = 300.0           # seconds
    use_mrv: bool = True                # Most Remaining Values heuristic
    use_degree: bool = True             # Degree heuristic
    use_forward_checking: bool = True   # Forward checking constraint propagation
    use_arc_consistency: bool = False   # AC-3 algorithm (expensive)
    word_score_weight: float = 0.7      # Weight for word quality scores
    crossing_weight: float = 0.3        # Weight for crossing potential
    min_word_score: float = 30.0        # Minimum acceptable word score
    theme_word_bonus: float = 20.0      # Bonus for theme words
    backtrack_limit: int = 1000         # Max backtracks per variable
    random_restarts: int = 0            # Number of random restarts
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.word_score_weight + self.crossing_weight != 1.0:
            # Normalize weights
            total = self.word_score_weight + self.crossing_weight
            self.word_score_weight /= total
            self.crossing_weight /= total


@dataclass
class SolverState:
    """Current state of the solver."""
    iterations: int = 0
    backtracks: int = 0
    forward_checks: int = 0
    constraint_checks: int = 0
    start_time: float = 0.0
    current_slot: Optional[GridSlot] = None
    domains: Dict[GridSlot, List[str]] = None
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = {}


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    slot1: GridSlot
    slot2: GridSlot
    position1: int
    position2: int
    letter1: str
    letter2: str
    message: str


class CrosswordConstraint:
    """Base class for crossword constraints."""
    
    def is_satisfied(self, grid: CrosswordGrid, slot: GridSlot, word: str) -> bool:
        """Check if placing word in slot satisfies this constraint."""
        raise NotImplementedError
    
    def get_violations(self, grid: CrosswordGrid) -> List[ConstraintViolation]:
        """Get all current violations of this constraint."""
        raise NotImplementedError


class CrossingConstraint(CrosswordConstraint):
    """Constraint that crossing words must have matching letters."""
    
    def is_satisfied(self, grid: CrosswordGrid, slot: GridSlot, word: str) -> bool:
        """Check if word satisfies crossing constraints."""
        crossing_slots = grid.get_crossing_slots(slot)
        
        for crossing_slot, (my_pos, crossing_pos) in crossing_slots:
            if crossing_slot.word:
                if word[my_pos] != crossing_slot.word[crossing_pos]:
                    return False
        
        return True
    
    def get_violations(self, grid: CrosswordGrid) -> List[ConstraintViolation]:
        """Get all crossing violations in the current grid."""
        violations = []
        
        for slot in grid.get_filled_slots():
            crossing_slots = grid.get_crossing_slots(slot)
            
            for crossing_slot, (my_pos, crossing_pos) in crossing_slots:
                if crossing_slot.word:
                    my_letter = slot.word[my_pos]
                    crossing_letter = crossing_slot.word[crossing_pos]
                    
                    if my_letter != crossing_letter:
                        violation = ConstraintViolation(
                            slot1=slot,
                            slot2=crossing_slot,
                            position1=my_pos,
                            position2=crossing_pos,
                            letter1=my_letter,
                            letter2=crossing_letter,
                            message=f"Letter mismatch: {my_letter} != {crossing_letter}"
                        )
                        violations.append(violation)
        
        return violations


class WordQualityConstraint(CrosswordConstraint):
    """Constraint that words must meet minimum quality standards."""
    
    def __init__(self, min_score: float = 30.0):
        self.min_score = min_score
    
    def is_satisfied(self, grid: CrosswordGrid, slot: GridSlot, word: str) -> bool:
        """Check if word meets quality standards."""
        # This would need access to lexicon for scoring
        # For now, implement basic quality checks
        
        # No repeated consecutive letters
        for i in range(len(word) - 1):
            if word[i] == word[i + 1]:
                return False
        
        # Must have at least one vowel
        if not any(c in 'AEIOU' for c in word):
            return False
        
        return True
    
    def get_violations(self, grid: CrosswordGrid) -> List[ConstraintViolation]:
        """Get quality violations (simplified)."""
        return []  # Would implement based on lexicon scoring


class CrosswordSolver:
    """Crossword puzzle solver using constraint satisfaction techniques."""
    
    def __init__(self, grid: CrosswordGrid, lexicon: Lexicon, config: SolverConfig):
        """Initialize the solver.
        
        Args:
            grid: The crossword grid to fill
            lexicon: Word lexicon for candidate words
            config: Solver configuration
        """
        self.grid = grid
        self.lexicon = lexicon
        self.config = config
        
        self.constraints: List[CrosswordConstraint] = [
            CrossingConstraint(),
            WordQualityConstraint(config.min_word_score)
        ]
        
        self.state = SolverState()
        self.construction_log: List[Dict[str, Any]] = []
        self.trace_file: Optional[str] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize domains
        self._initialize_domains()
    
    def _initialize_domains(self):
        """Initialize the domains for all slots."""
        self.state.domains = {}
        
        for slot in self.grid.get_unfilled_slots():
            domain = self._get_initial_domain(slot)
            self.state.domains[slot] = domain
            
            self.logger.debug(f"Initialized domain for slot {slot.start_row},{slot.start_col} "
                            f"{slot.direction.value}: {len(domain)} words")
    
    def _get_initial_domain(self, slot: GridSlot) -> List[str]:
        """Get the initial domain of possible words for a slot."""
        # Get words of correct length
        candidates = self.lexicon.get_words_by_length(slot.length)
        
        # Filter by crossing constraints
        crossing_constraints = {}
        crossing_slots = self.grid.get_crossing_slots(slot)
        
        for crossing_slot, (my_pos, crossing_pos) in crossing_slots:
            if crossing_slot.word:
                crossing_constraints[my_pos] = crossing_slot.word[crossing_pos]
        
        # Filter candidates
        if crossing_constraints:
            filtered_candidates = self.lexicon.get_words_with_constraints(
                slot.length, crossing_constraints, self.config.min_word_score
            )
        else:
            filtered_candidates = [w for w in candidates 
                                 if self.lexicon.get_word_score(w) >= self.config.min_word_score]
        
        # Sort by heuristic score
        scored_candidates = [(word, self._calculate_word_score(slot, word)) 
                           for word in filtered_candidates]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, score in scored_candidates]
    
    def _calculate_word_score(self, slot: GridSlot, word: str) -> float:
        """Calculate a heuristic score for placing a word in a slot."""
        base_score = self.lexicon.get_word_score(word)
        
        # Crossing potential score
        crossing_slots = self.grid.get_crossing_slots(slot)
        crossing_score = 0.0
        
        for crossing_slot, (my_pos, crossing_pos) in crossing_slots:
            if not crossing_slot.word:
                # Count how many words can cross at this position
                crossing_letter = word[my_pos]
                crossing_candidates = [w for w in self.state.domains.get(crossing_slot, [])
                                     if w[crossing_pos] == crossing_letter]
                crossing_score += len(crossing_candidates)
        
        # Normalize crossing score
        if crossing_slots:
            crossing_score /= len(crossing_slots)
        
        # Combined score
        final_score = (self.config.word_score_weight * base_score + 
                      self.config.crossing_weight * crossing_score)
        
        return final_score
    
    def solve(self) -> bool:
        """Solve the crossword puzzle."""
        self.state.start_time = time.time()
        self.logger.info("Starting crossword solving...")
        
        # Handle theme entries first if any
        if self.grid.theme_entries:
            self.grid.place_theme_entries()
            self._update_domains_after_placement()
        
        # Main solving loop
        success = self._backtrack_search()
        
        elapsed_time = time.time() - self.state.start_time
        
        self.logger.info(f"Solving completed in {elapsed_time:.2f}s")
        self.logger.info(f"Iterations: {self.state.iterations}")
        self.logger.info(f"Backtracks: {self.state.backtracks}")
        self.logger.info(f"Forward checks: {self.state.forward_checks}")
        self.logger.info(f"Success: {success}")
        
        if success:
            self._log_construction_step("SOLUTION_FOUND", {
                'total_time': elapsed_time,
                'iterations': self.state.iterations,
                'backtracks': self.state.backtracks,
                'fill_percentage': self.grid.get_fill_percentage()
            })
        
        return success
    
    def _backtrack_search(self) -> bool:
        """Main backtracking search algorithm."""
        if self._is_complete():
            return True
        
        if self._should_terminate():
            return False
        
        # Select next variable (slot) to fill
        slot = self._select_unassigned_variable()
        if not slot:
            return True
        
        self.state.current_slot = slot
        self.state.iterations += 1
        
        # Try each value in the domain
        domain = self.state.domains.get(slot, [])
        backtrack_count = 0
        
        for word in domain:
            if backtrack_count >= self.config.backtrack_limit:
                break
            
            if self._is_consistent(slot, word):
                # Make assignment
                old_state = self._save_state()
                self._assign_word(slot, word)
                
                self._log_construction_step("ASSIGN", {
                    'slot': f"{slot.start_row},{slot.start_col} {slot.direction.value}",
                    'word': word,
                    'score': self.lexicon.get_word_score(word)
                })
                
                # Forward checking
                if self.config.use_forward_checking:
                    if not self._forward_check(slot, word):
                        self._restore_state(old_state)
                        backtrack_count += 1
                        continue
                
                # Recursive call
                if self._backtrack_search():
                    return True
                
                # Backtrack
                self._restore_state(old_state)
                self.state.backtracks += 1
                backtrack_count += 1
                
                self._log_construction_step("BACKTRACK", {
                    'slot': f"{slot.start_row},{slot.start_col} {slot.direction.value}",
                    'word': word,
                    'backtrack_count': backtrack_count
                })
        
        return False
    
    def _is_complete(self) -> bool:
        """Check if the puzzle is completely solved."""
        return self.grid.is_complete()
    
    def _should_terminate(self) -> bool:
        """Check if solver should terminate due to limits."""
        if self.state.iterations >= self.config.max_iterations:
            self.logger.warning(f"Reached iteration limit: {self.config.max_iterations}")
            return True
        
        elapsed_time = time.time() - self.state.start_time
        if elapsed_time >= self.config.time_limit:
            self.logger.warning(f"Reached time limit: {self.config.time_limit}s")
            return True
        
        return False
    
    def _select_unassigned_variable(self) -> Optional[GridSlot]:
        """Select the next unfilled slot using heuristics."""
        unfilled_slots = self.grid.get_unfilled_slots()
        if not unfilled_slots:
            return None
        
        if self.config.use_mrv and self.config.use_degree:
            return self._select_mrv_degree(unfilled_slots)
        elif self.config.use_mrv:
            return self._select_mrv(unfilled_slots)
        elif self.config.use_degree:
            return self._select_degree(unfilled_slots)
        else:
            return random.choice(unfilled_slots)
    
    def _select_mrv(self, slots: List[GridSlot]) -> GridSlot:
        """Select slot with Most Remaining Values (smallest domain)."""
        return min(slots, key=lambda s: len(self.state.domains.get(s, [])))
    
    def _select_degree(self, slots: List[GridSlot]) -> GridSlot:
        """Select slot with highest degree (most constraints)."""
        def degree(slot):
            return len([cs for cs, _ in self.grid.get_crossing_slots(slot) if not cs.word])
        
        return max(slots, key=degree)
    
    def _select_mrv_degree(self, slots: List[GridSlot]) -> GridSlot:
        """Select using MRV, breaking ties with degree heuristic."""
        min_domain_size = min(len(self.state.domains.get(s, [])) for s in slots)
        mrv_slots = [s for s in slots if len(self.state.domains.get(s, [])) == min_domain_size]
        
        if len(mrv_slots) == 1:
            return mrv_slots[0]
        
        return self._select_degree(mrv_slots)
    
    def _is_consistent(self, slot: GridSlot, word: str) -> bool:
        """Check if assigning word to slot is consistent with constraints."""
        self.state.constraint_checks += 1
        
        for constraint in self.constraints:
            if not constraint.is_satisfied(self.grid, slot, word):
                return False
        
        return True
    
    def _assign_word(self, slot: GridSlot, word: str):
        """Assign a word to a slot."""
        self.grid.place_word(slot, word)
        
        # Remove slot from unfilled domains
        if slot in self.state.domains:
            del self.state.domains[slot]
    
    def _forward_check(self, slot: GridSlot, word: str) -> bool:
        """Perform forward checking to prune domains."""
        self.state.forward_checks += 1
        
        # Check each crossing slot
        crossing_slots = self.grid.get_crossing_slots(slot)
        
        for crossing_slot, (my_pos, crossing_pos) in crossing_slots:
            if crossing_slot.word or crossing_slot not in self.state.domains:
                continue
            
            crossing_letter = word[my_pos]
            
            # Filter domain to only include words with correct crossing letter
            old_domain = self.state.domains[crossing_slot]
            new_domain = [w for w in old_domain if w[crossing_pos] == crossing_letter]
            
            self.state.domains[crossing_slot] = new_domain
            
            # If domain becomes empty, forward checking fails
            if not new_domain:
                return False
        
        return True
    
    def _save_state(self) -> Dict[str, Any]:
        """Save current solver state for backtracking."""
        return {
            'grid_state': self.grid.clone(),
            'domains': {slot: domain[:] for slot, domain in self.state.domains.items()}
        }
    
    def _restore_state(self, saved_state: Dict[str, Any]):
        """Restore solver state during backtracking."""
        self.grid = saved_state['grid_state']
        self.state.domains = saved_state['domains']
    
    def _update_domains_after_placement(self):
        """Update all domains after placing theme entries."""
        for slot in self.grid.get_unfilled_slots():
            self.state.domains[slot] = self._get_initial_domain(slot)
    
    def _log_construction_step(self, action: str, details: Dict[str, Any]):
        """Log a construction step."""
        step = {
            'iteration': self.state.iterations,
            'action': action,
            'timestamp': time.time() - self.state.start_time,
            'details': details
        }
        self.construction_log.append(step)
        
        if self.trace_file:
            self._write_trace_step(step)
    
    def _write_trace_step(self, step: Dict[str, Any]):
        """Write a trace step to file."""
        try:
            with open(self.trace_file, 'a', encoding='utf-8') as f:
                import json
                f.write(json.dumps(step) + '\n')
        except Exception as e:
            self.logger.error(f"Error writing trace: {e}")
    
    def enable_tracing(self, trace_file: str):
        """Enable detailed tracing to file."""
        self.trace_file = trace_file
        
        # Initialize trace file with metadata
        try:
            with open(trace_file, 'w', encoding='utf-8') as f:
                import json
                metadata = {
                    'grid_size': f"{self.grid.width}x{self.grid.height}",
                    'symmetry': self.grid.symmetry.value,
                    'config': {
                        'max_iterations': self.config.max_iterations,
                        'time_limit': self.config.time_limit,
                        'use_mrv': self.config.use_mrv,
                        'use_degree': self.config.use_degree,
                        'use_forward_checking': self.config.use_forward_checking
                    },
                    'lexicon_size': len(self.lexicon),
                    'total_slots': len(self.grid.get_all_slots())
                }
                f.write(json.dumps({'type': 'METADATA', 'data': metadata}) + '\n')
        except Exception as e:
            self.logger.error(f"Error initializing trace file: {e}")
    
    def get_construction_log(self) -> List[Dict[str, Any]]:
        """Get the construction log."""
        return self.construction_log
    
    def get_solver_statistics(self) -> Dict[str, Any]:
        """Get solver performance statistics."""
        elapsed_time = time.time() - self.state.start_time
        
        return {
            'iterations': self.state.iterations,
            'backtracks': self.state.backtracks,
            'forward_checks': self.state.forward_checks,
            'constraint_checks': self.state.constraint_checks,
            'elapsed_time': elapsed_time,
            'iterations_per_second': self.state.iterations / max(elapsed_time, 0.001),
            'backtrack_ratio': self.state.backtracks / max(self.state.iterations, 1),
            'fill_percentage': self.grid.get_fill_percentage(),
            'is_complete': self._is_complete()
        }


class AdvancedSolver(CrosswordSolver):
    """Advanced solver with additional heuristics and optimizations."""
    
    def __init__(self, grid: CrosswordGrid, lexicon: Lexicon, config: SolverConfig):
        super().__init__(grid, lexicon, config)
        
        # Additional state for advanced features
        self.failed_assignments: Dict[GridSlot, Set[str]] = defaultdict(set)
        self.slot_difficulty: Dict[GridSlot, float] = {}
        
        # Calculate initial slot difficulties
        self._calculate_slot_difficulties()
    
    def _calculate_slot_difficulties(self):
        """Calculate difficulty scores for each slot."""
        for slot in self.grid.get_all_slots():
            difficulty = self._calculate_slot_difficulty(slot)
            self.slot_difficulty[slot] = difficulty
    
    def _calculate_slot_difficulty(self, slot: GridSlot) -> float:
        """Calculate how difficult a slot is to fill."""
        # Base difficulty from domain size
        domain_size = len(self.state.domains.get(slot, []))
        if domain_size == 0:
            return float('inf')
        
        base_difficulty = 1.0 / domain_size
        
        # Crossing complexity
        crossing_slots = self.grid.get_crossing_slots(slot)
        crossing_difficulty = len(crossing_slots) * 0.1
        
        # Length factor (very short or very long words are harder)
        length_difficulty = 0
        if slot.length <= 3:
            length_difficulty = 0.3
        elif slot.length >= 12:
            length_difficulty = 0.2
        
        return base_difficulty + crossing_difficulty + length_difficulty
    
    def _select_unassigned_variable(self) -> Optional[GridSlot]:
        """Enhanced variable selection using multiple heuristics."""
        unfilled_slots = self.grid.get_unfilled_slots()
        if not unfilled_slots:
            return None
        
        # Filter out slots with empty domains
        viable_slots = [s for s in unfilled_slots if self.state.domains.get(s, [])]
        if not viable_slots:
            return None
        
        # Use composite scoring
        scored_slots = []
        for slot in viable_slots:
            score = self._calculate_variable_selection_score(slot)
            scored_slots.append((slot, score))
        
        # Sort by score (higher is better for selection)
        scored_slots.sort(key=lambda x: x[1], reverse=True)
        
        return scored_slots[0][0]
    
    def _calculate_variable_selection_score(self, slot: GridSlot) -> float:
        """Calculate composite score for variable selection."""
        # MRV component (inverse of domain size)
        domain_size = len(self.state.domains.get(slot, []))
        mrv_score = 1.0 / max(domain_size, 1)
        
        # Degree component
        unfilled_crossings = sum(1 for cs, _ in self.grid.get_crossing_slots(slot) if not cs.word)
        degree_score = unfilled_crossings
        
        # Difficulty component
        difficulty_score = self.slot_difficulty.get(slot, 0)
        
        # Failed assignment penalty
        failed_count = len(self.failed_assignments.get(slot, set()))
        failure_penalty = failed_count * 0.1
        
        # Combined score
        composite_score = (mrv_score * 0.4 + 
                          degree_score * 0.3 + 
                          difficulty_score * 0.2 - 
                          failure_penalty * 0.1)
        
        return composite_score
    
    def _order_domain_values(self, slot: GridSlot) -> List[str]:
        """Order domain values using advanced heuristics."""
        domain = self.state.domains.get(slot, [])
        if not domain:
            return []
        
        # Score each word
        scored_words = []
        for word in domain:
            if word in self.failed_assignments.get(slot, set()):
                continue  # Skip previously failed assignments
            
            score = self._calculate_value_ordering_score(slot, word)
            scored_words.append((word, score))
        
        # Sort by score (descending)
        scored_words.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, score in scored_words]
    
    def _calculate_value_ordering_score(self, slot: GridSlot, word: str) -> float:
        """Calculate score for value ordering heuristic."""
        # Base word quality score
        base_score = self.lexicon.get_word_score(word)
        
        # Least constraining value: count how many options remain for crossing slots
        constraining_score = 0
        crossing_slots = self.grid.get_crossing_slots(slot)
        
        for crossing_slot, (my_pos, crossing_pos) in crossing_slots:
            if crossing_slot.word or crossing_slot not in self.state.domains:
                continue
            
            crossing_letter = word[my_pos]
            compatible_words = [w for w in self.state.domains[crossing_slot] 
                              if w[crossing_pos] == crossing_letter]
            constraining_score += len(compatible_words)
        
        # Normalize
        if crossing_slots:
            constraining_score /= len(crossing_slots)
        
        # Combine scores
        return base_score * 0.6 + constraining_score * 0.4
    
    def _backtrack_search(self) -> bool:
        """Enhanced backtracking with learning."""
        if self._is_complete():
            return True
        
        if self._should_terminate():
            return False
        
        slot = self._select_unassigned_variable()
        if not slot:
            return True
        
        self.state.current_slot = slot
        self.state.iterations += 1
        
        # Use ordered domain values
        ordered_domain = self._order_domain_values(slot)
        
        for word in ordered_domain:
            if self._is_consistent(slot, word):
                old_state = self._save_state()
                self._assign_word(slot, word)
                
                # Enhanced forward checking
                if self.config.use_forward_checking:
                    if not self._enhanced_forward_check(slot, word):
                        self._restore_state(old_state)
                        self.failed_assignments[slot].add(word)
                        continue
                
                if self._backtrack_search():
                    return True
                
                # Learn from failure
                self._restore_state(old_state)
                self.failed_assignments[slot].add(word)
                self.state.backtracks += 1
        
        return False
    
    def _enhanced_forward_check(self, slot: GridSlot, word: str) -> bool:
        """Enhanced forward checking with arc consistency."""
        if not super()._forward_check(slot, word):
            return False
        
        # Optional: Apply arc consistency (AC-3)
        if self.config.use_arc_consistency:
            return self._apply_arc_consistency()
        
        return True
    
    def _apply_arc_consistency(self) -> bool:
        """Apply AC-3 algorithm for arc consistency."""
        # Build constraint graph
        arcs = []
        
        for slot in self.state.domains:
            crossing_slots = self.grid.get_crossing_slots(slot)
            for crossing_slot, positions in crossing_slots:
                if crossing_slot in self.state.domains:
                    arcs.append((slot, crossing_slot, positions))
        
        # AC-3 algorithm
        while arcs:
            slot1, slot2, (pos1, pos2) = arcs.pop(0)
            
            if self._revise_domain(slot1, slot2, pos1, pos2):
                if not self.state.domains[slot1]:
                    return False  # Domain became empty
                
                # Add arcs from neighbors of slot1
                for neighbor_slot, neighbor_positions in self.grid.get_crossing_slots(slot1):
                    if neighbor_slot in self.state.domains and neighbor_slot != slot2:
                        arcs.append((neighbor_slot, slot1, (neighbor_positions[1], neighbor_positions[0])))
        
        return True
    
    def _revise_domain(self, slot1: GridSlot, slot2: GridSlot, pos1: int, pos2: int) -> bool:
        """Revise domain of slot1 with respect to slot2."""
        revised = False
        domain1 = self.state.domains[slot1][:]
        
        for word1 in domain1:
            letter1 = word1[pos1]
            
            # Check if there exists a compatible word in slot2's domain
            compatible = any(word2[pos2] == letter1 for word2 in self.state.domains[slot2])
            
            if not compatible:
                self.state.domains[slot1].remove(word1)
                revised = True
        
        return revised


class SolverFactory:
    """Factory for creating crossword solvers."""
    
    @staticmethod
    def create_solver(solver_type: str, grid: CrosswordGrid, 
                     lexicon: Lexicon, config: SolverConfig) -> CrosswordSolver:
        """Create a solver of the specified type."""
        if solver_type.lower() == 'basic':
            return CrosswordSolver(grid, lexicon, config)
        elif solver_type.lower() == 'advanced':
            return AdvancedSolver(grid, lexicon, config)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
    
    @staticmethod
    def create_config(**kwargs) -> SolverConfig:
        """Create a solver configuration with custom parameters."""
        return SolverConfig(**kwargs)


def solve_crossword(grid: CrosswordGrid, lexicon: Lexicon, 
                   solver_type: str = 'advanced', **config_kwargs) -> Tuple[bool, CrosswordSolver]:
    """Convenience function to solve a crossword puzzle."""
    config = SolverFactory.create_config(**config_kwargs)
    solver = SolverFactory.create_solver(solver_type, grid, lexicon, config)
    
    success = solver.solve()
    return success, solver


# Utility functions for solver analysis and debugging

def analyze_solver_performance(solver: CrosswordSolver) -> Dict[str, Any]:
    """Analyze solver performance and provide insights."""
    stats = solver.get_solver_statistics()
    
    analysis = {
        'performance_metrics': stats,
        'efficiency': {
            'backtrack_efficiency': 1.0 - stats['backtrack_ratio'],
            'time_efficiency': stats['iterations_per_second'],
            'space_efficiency': stats['fill_percentage']
        },
        'bottlenecks': [],
        'recommendations': []
    }
    
    # Identify bottlenecks
    if stats['backtrack_ratio'] > 0.5:
        analysis['bottlenecks'].append('High backtrack ratio - consider better heuristics')
    
    if stats['iterations_per_second'] < 100:
        analysis['bottlenecks'].append('Low iteration rate - check constraint checking efficiency')
    
    if stats['fill_percentage'] < 80:
        analysis['bottlenecks'].append('Low fill rate - may need more flexible constraints or better wordlist')
    
    # Generate recommendations
    if len(analysis['bottlenecks']) > 0:
        analysis['recommendations'].append('Consider using advanced solver with learning')
        analysis['recommendations'].append('Increase time limit or iteration count')
        analysis['recommendations'].append('Review wordlist quality and coverage')
    
    return analysis


def debug_unsatisfiable_grid(grid: CrosswordGrid, lexicon: Lexicon) -> Dict[str, Any]:
    """Debug why a grid might be unsatisfiable."""
    debug_info = {
        'grid_metrics': {
            'size': f"{grid.width}x{grid.height}",
            'total_slots': len(grid.get_all_slots()),
            'across_slots': len(grid.across_slots),
            'down_slots': len(grid.down_slots),
            'theme_entries': len(grid.theme_entries)
        },
        'lexicon_coverage': {},
        'problematic_slots': [],
        'constraint_analysis': {}
    }
    
    # Analyze lexicon coverage
    length_coverage = {}
    for slot in grid.get_all_slots():
        length = slot.length
        available_words = len(lexicon.get_words_by_length(length))
        length_coverage[length] = available_words
        
        if available_words < 10:  # Threshold for problematic coverage
            debug_info['problematic_slots'].append({
                'slot': f"{slot.start_row},{slot.start_col} {slot.direction.value}",
                'length': length,
                'available_words': available_words
            })
    
    debug_info['lexicon_coverage'] = length_coverage
    
    # Analyze constraints
    crossing_constraint = CrossingConstraint()
    violations = crossing_constraint.get_violations(grid)
    
    debug_info['constraint_analysis'] = {
        'crossing_violations': len(violations),
        'violations': [
            {
                'slot1': f"{v.slot1.start_row},{v.slot1.start_col} {v.slot1.direction.value}",
                'slot2': f"{v.slot2.start_row},{v.slot2.start_col} {v.slot2.direction.value}",
                'message': v.message
            }
            for v in violations[:10]  # Limit to first 10
        ]
    }
    
    return debug_info
