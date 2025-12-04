#!/usr/bin/env python3
"""
Local evaluation script for chess agents.

This script evaluates a chess agent by playing games against multiple opponents
(Stockfish depth 1 and Random agent) and provides
ACPL scores, win/draw/loss rates, and average move times.
"""

import os
import re
import sys
import time
import json
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from jinja2 import Template

import chess

# Add chess-env to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "chess-env"))

from agents import ChessAgent, RandomAgent
from env import ChessEnvironment
from run_game import _StockfishAnalyzer
from chess_renderer import ChessRenderer


def render_template(template_name: str, **kwargs) -> str:
    """
    Render a Jinja2 template with the given variables.
    
    Args:
        template_name: Name of the template file (relative to script directory)
        **kwargs: Variables to pass to the template
    
    Returns:
        Rendered template string
    
    Raises:
        SystemExit: If template file not found or rendering fails
    """
    template_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(template_dir, template_name)
    
    if not os.path.exists(template_path):
        print(f"Error: Template file not found: {template_path}")
        sys.exit(1)
    
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        template = Template(template_content)
        return template.render(**kwargs)
    except Exception as e:
        print(f"Error rendering template '{template_name}': {e}")
        sys.exit(1)


@dataclass
class GameStats:
    """Statistics for a single game."""
    result: str
    moves_played: int
    white_time: float
    black_time: float
    white_acpl: float
    black_acpl: float
    player_color: str  # "white" or "black"


@dataclass
class EvaluationResults:
    """Overall evaluation results."""
    opponent_name: str
    total_games: int
    wins: int
    draws: int
    losses: int
    avg_acpl: float
    avg_time_per_move: float
    games: List[GameStats]


class OpenAIEndpointAgent(ChessAgent):
    """
    Chess agent that calls an OpenAI-compatible API endpoint.
    
    This agent sends requests to a local or remote endpoint that implements
    the OpenAI chat completions API format.
    """
    
    # Unicode chess piece characters (same as openai_agent.py)
    UNICODE_PIECES = {
        'P': '♙',  # White pawn
        'R': '♖',  # White rook
        'N': '♘',  # White knight
        'B': '♗',  # White bishop
        'Q': '♕',  # White queen
        'K': '♔',  # White king
        
        'p': '♟',  # Black pawn
        'r': '♜',  # Black rook
        'n': '♞',  # Black knight
        'b': '♝',  # Black bishop
        'q': '♛',  # Black queen
        'k': '♚',  # Black king
    }
    
    def __init__(self, base_url: str, api_key: str = "dummy", max_retries: int = 2, model: str = "aicrowd-chess-model", 
                 template_file: Optional[str] = None, debug: bool = False):
        """
        Initialize the OpenAI endpoint agent.
        
        Args:
            base_url: Base URL of the OpenAI-compatible API endpoint
            api_key: API key (default: "dummy" for local servers)
            max_retries: Maximum number of retries for illegal moves before resigning
            model: Model name to use for API calls (default: "aicrowd-chess-model")
            template_file: Path to Jinja2 template file for prompt formatting (optional)
            debug: If True, print prompts and responses for debugging
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_retries = max_retries
        self.model = model
        self.template_file = template_file
        self.debug = debug
        self.move_times = []  # Track time for each move
        # Initialize chess renderer for board ASCII representation
        self.renderer = ChessRenderer(show_coordinates=True, show_move_numbers=False, 
                                      empty_square_char="·", use_rich=False)
    
    def _render_board_unicode(self, board: chess.Board) -> str:
        """
        Render the chess board using Unicode chess pieces.
        
        Args:
            board: The chess board to render
            
        Returns:
            String representation of the board with Unicode pieces
        """
        lines = []
        
        # Board coordinates
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        
        # Add top coordinate line with proper alignment
        # Each square is 3 characters wide, so we need to center each letter
        coord_parts = []
        for file in files:
            coord_parts.append(f" {file} ")  # 3-character spacing to match board squares
        coord_line = "   " + "".join(coord_parts) + "  "
        lines.append(coord_line)
        # Calculate border width: 8 squares × 3 characters each = 24 characters
        border_width = len(files) * 3
        lines.append("   +" + "-" * border_width + "+")
        
        # Render board squares
        for rank_idx, rank in enumerate(ranks):
            line_parts = []
            
            # Add rank coordinate
            line_parts.append(f"{rank} |")
            
            # Add squares
            for file_idx, file in enumerate(files):
                square = chess.parse_square(file + rank)
                piece = board.piece_at(square)
                
                # Get piece symbol or empty square character
                if piece is None:
                    piece_char = "·"  # Empty square
                else:
                    piece_char = self.UNICODE_PIECES[piece.symbol()]
                
                # Format square
                square_str = f" {piece_char} "
                line_parts.append(square_str)
            
            # Add closing coordinate
            line_parts.append(f"| {rank}")
            lines.append("".join(line_parts))
        
        # Add bottom coordinate line
        lines.append("   +" + "-" * border_width + "+")
        coord_line = "   " + "".join(coord_parts) + "  "
        lines.append(coord_line)
        
        return "\n".join(lines)
    
    def _build_prompt_context(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> dict:
        """
        Build the context dictionary for Jinja2 template rendering.
        
        Args:
            board: Current chess board state
            legal_moves: List of legal moves available
            move_history: List of moves played so far (in UCI notation)
            side_to_move: Which side is to move ('White' or 'Black')
            
        Returns:
            Dictionary with all template variables (both string and list forms)
        """
        # Get FEN representation
        fen = board.fen()
        
        # Get UTF board representation with Unicode chess pieces
        board_utf = self._render_board_unicode(board)
        
        # Get ASCII board representation
        board_ascii = board.unicode()
        
        # Get last move description
        if board.move_stack:
            last_move = board.move_stack[-1]
            # We need to get the SAN before the move was made
            # Create a temporary board to get the SAN
            temp_board = chess.Board()
            for move in board.move_stack[:-1]:
                temp_board.push(move)
            last_move_san = temp_board.san(last_move)
            last_side = "Black" if board.turn else "White"
            last_move_desc = f"{last_side} played {last_move_san}"
        else:
            last_move_desc = "(start of game)"
        
        # Format legal moves as both UCI and SAN lists
        legal_moves_uci_list = [move.uci() for move in legal_moves]
        legal_moves_san_list = [board.san(move) for move in legal_moves]
        legal_moves_uci_str = " ".join(legal_moves_uci_list)
        legal_moves_san_str = " ".join(legal_moves_san_list)
        
        # Format move history as both UCI and SAN
        if move_history:
            # Move history is already in UCI format
            move_history_uci_list = list(move_history)
            move_history_uci_str = " ".join(move_history_uci_list)
            
            # Convert UCI moves to SAN if possible
            try:
                history_board = chess.Board()
                move_history_san_list = []
                for uci_move in move_history:
                    try:
                        move = chess.Move.from_uci(uci_move)
                        san = history_board.san(move)
                        move_history_san_list.append(san)
                        history_board.push(move)
                    except Exception:
                        move_history_san_list.append(uci_move)
                
                move_history_san_str = " ".join(move_history_san_list)
            except Exception:
                move_history_san_list = list(move_history)
                move_history_san_str = " ".join(move_history)
        else:
            move_history_uci_list = []
            move_history_san_list = []
            move_history_uci_str = "(no moves yet)"
            move_history_san_str = "(no moves yet)"
        
        # Get first legal move as an example
        first_legal_move = legal_moves_uci_list[0] if legal_moves_uci_list else ""
        
        # Build and return the context dictionary
        return {
            "board_utf": board_utf,
            "board_ascii": board_ascii,
            "FEN": fen,
            "side_to_move": side_to_move,
            "last_move": last_move_desc,
            "legal_moves_uci": legal_moves_uci_str,
            "legal_moves_san": legal_moves_san_str,
            "move_history_uci": move_history_uci_str,
            "move_history_san": move_history_san_str,
            "legal_moves_uci_list": legal_moves_uci_list,
            "legal_moves_san_list": legal_moves_san_list,
            "move_history_uci_list": move_history_uci_list,
            "move_history_san_list": move_history_san_list,
            "first_legal_move": first_legal_move,
        }
    
    def _format_prompt(self, board: chess.Board, legal_moves: List[chess.Move], 
                      move_history: List[str], side_to_move: str) -> str:
        """Format the prompt for the API call."""
        # Build the context using the shared method (same as openai_agent.py)
        context = self._build_prompt_context(board, legal_moves, move_history, side_to_move)
        
        # If template file is specified, use it
        if self.template_file:
            prompt = render_template(
                self.template_file,
                **context
            )
        else:
            raise FileNotFoundError(f"Template file not found: {self.template_file}")
        
        return prompt
    
    def _parse_move(self, response: str, legal_moves: List[chess.Move]) -> Optional[chess.Move]:
        """Parse the move from the API response."""
        # Extract move from <uci_move> tags
        match = re.search(r'<uci_move>(.*?)</uci_move>', response, re.IGNORECASE | re.DOTALL)
        if not match:
            print(f"Warning: Could not find <uci_move> tags in response: {response[:100]}")
            return None
        
        move_str = match.group(1).strip()
        
        # Check for resignation
        if move_str.lower() == "resign":
            return None
        
        try:
            move = chess.Move.from_uci(move_str)
            if move in legal_moves:
                return move
            else:
                print(f"Warning: Parsed move {move_str} is not in legal moves")
                return None
        except FileNotFoundError:
            print(f"Warning: Template file not found: {self.template_file}")
            import sys; sys.exit(1)
        except Exception as e:
            print(f"Warning: Failed to parse move '{move_str}': {e}")
            return None
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> Tuple[Optional[chess.Move], Optional[str]]:
        """
        Choose a move by calling the OpenAI-compatible API endpoint.
        
        Implements retry logic: tries up to max_retries times for illegal moves,
        then resigns.
        """
        if not legal_moves:
            return None, "No legal moves available"
        
        for attempt in range(self.max_retries + 1):
            try:
                # Format prompt
                prompt = self._format_prompt(board, legal_moves, move_history, side_to_move)
                
                # Debug: Print input prompt
                if self.debug:
                    print(f"\n{'='*70}")
                    print(f"DEBUG - INPUT PROMPT (Attempt {attempt + 1}):")
                    print(f"{'='*70}")
                    print(prompt)
                    print(f"{'='*70}\n")
                
                # Call API
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,

                )
                elapsed_time = time.time() - start_time
                self.move_times.append(elapsed_time)
                
                # Extract response
                if not response.choices:
                    print(f"Warning: Empty response from API (attempt {attempt + 1}/{self.max_retries + 1})")
                    continue
                
                content = response.choices[0].message.content
                
                # Debug: Print output response
                if self.debug:
                    print(f"\n{'='*70}")
                    print(f"DEBUG - OUTPUT RESPONSE (Attempt {attempt + 1}, Time: {elapsed_time:.2f}s):")
                    print(f"{'='*70}")
                    print(content)
                    print(f"{'='*70}\n")
                
                # Parse move
                move = self._parse_move(content, legal_moves)
                
                if move is not None:
                    return move, f"API move (attempt {attempt + 1})"
                elif attempt < self.max_retries:
                    print(f"Warning: Invalid move on attempt {attempt + 1}, retrying...")
                    continue
                else:
                    print(f"Warning: Failed after {self.max_retries + 1} attempts, resigning")
                    return None, f"Resigned after {self.max_retries + 1} failed attempts"
                
            except Exception as e:
                if attempt < self.max_retries:
                    print(f"Warning: API call failed on attempt {attempt + 1}: {e}, retrying...")
                    continue
                else:
                    print(f"Error: API call failed after {self.max_retries + 1} attempts: {e}")
                    return None, f"Resigned due to API error: {e}"
        
        return None, "Failed to get valid move"
    
    def get_avg_move_time(self) -> float:
        """Get average time per move."""
        return sum(self.move_times) / len(self.move_times) if self.move_times else 0.0
    
    def reset_stats(self):
        """Reset move time statistics."""
        self.move_times = []


class StockfishAgent(ChessAgent):
    """Simple Stockfish agent using python-chess engine."""
    
    def __init__(self, depth: int = 1, skill_level: int = 0, time_limit_ms: int = 100):
        """
        Initialize Stockfish agent.
        
        Args:
            depth: Search depth for Stockfish
            skill_level: Skill level (0-20, lower is weaker)
            time_limit_ms: Time limit in milliseconds
        """
        self.depth = depth
        self.skill_level = skill_level
        self.time_limit_ms = time_limit_ms
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        if skill_level is not None:
            self.engine.configure({"Skill Level": skill_level})
    
    def choose_move(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        move_history: List[str],
        side_to_move: str,
    ) -> Tuple[Optional[chess.Move], Optional[str]]:
        """Choose a move using Stockfish."""
        if not legal_moves:
            return None, "No legal moves available"
        
        try:
            result = self.engine.play(
                board,
                chess.engine.Limit(depth=self.depth, time=self.time_limit_ms / 1000.0)
            )
            return result.move, "Stockfish move"
        except Exception as e:
            print(f"Stockfish error: {e}")
            return None, f"Stockfish error: {e}"
    
    def close(self):
        """Close the Stockfish engine."""
        if hasattr(self, 'engine'):
            self.engine.quit()


def play_game(player_agent: OpenAIEndpointAgent, opponent_agent: ChessAgent,
              player_color: str, game_id: int, verbose: bool = False) -> dict:
    """
    Play a single game between player and opponent.
    
    Args:
        player_agent: The player agent being evaluated
        opponent_agent: The opponent agent
        player_color: "white" or "black"
        game_id: Game number (for logging)
        verbose: Whether to print game progress
    
    Returns:
        Dictionary with game statistics
    """
    # Set up agents based on colors
    if player_color == "white":
        white_agent = player_agent
        black_agent = opponent_agent
    else:
        white_agent = opponent_agent
        black_agent = player_agent
    
    # Create environment
    env = ChessEnvironment(white_agent, black_agent, max_moves=200, time_limit=30.0)
    
    # Track move times
    white_times = []
    black_times = []
    
    # Play the game
    if verbose:
        print(f"\n{'='*60}")
        print(f"Game {game_id}: Player as {player_color.upper()} vs {opponent_agent.__class__.__name__}")
        print(f"{'='*60}")
    
    result = env.play_game(verbose=verbose)
    
    # Extract move times from player agent
    player_times = player_agent.move_times.copy()
    player_agent.reset_stats()
    
    # Calculate times
    if player_color == "white":
        white_time = sum(player_times) / len(player_times) if player_times else 0.0
        black_time = 0.0  # Opponent time not tracked
    else:
        white_time = 0.0
        black_time = sum(player_times) / len(player_times) if player_times else 0.0
    
    return {
        "result": result["result"],
        "moves_played": result["moves_played"],
        "move_history": result["move_history"],
        "white_time": white_time,
        "black_time": black_time
    }


def save_game_log(
    game_num: int,
    opponent_name: str,
    player_color: str,
    game_result: dict,
    white_acpl: float,
    black_acpl: float,
    timestamp: str
):
    """
    Save game data to a JSON file in the logs/ directory.
    
    Args:
        game_num: Game number
        opponent_name: Name of the opponent
        player_color: "white" or "black"
        game_result: Dictionary with game result data
        white_acpl: White's average centipawn loss
        black_acpl: Black's average centipawn loss
        timestamp: Timestamp string for the log filename
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Determine player ACPL based on color
    player_acpl = white_acpl if player_color == "white" else black_acpl
    opponent_acpl = black_acpl if player_color == "white" else white_acpl
    
    # Create game data structure
    game_data = {
        "timestamp": timestamp,
        "game_number": game_num,
        "opponent": opponent_name,
        "player_color": player_color,
        "result": game_result["result"],
        "moves_played": game_result["moves_played"],
        "move_history": game_result["move_history"],
        "player_acpl": player_acpl,
        "opponent_acpl": opponent_acpl,
        "white_acpl": white_acpl,
        "black_acpl": black_acpl,
        "player_avg_time_per_move": game_result.get("white_time" if player_color == "white" else "black_time", 0.0)
    }
    
    # Generate filename with timestamp and game number
    # Safe opponent name (replace spaces and special chars)
    safe_opponent = opponent_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    filename = f"game_{timestamp}_{safe_opponent}_{player_color}_g{game_num}.json"
    filepath = os.path.join(logs_dir, filename)
    
    # Write JSON file
    with open(filepath, 'w') as f:
        json.dump(game_data, f, indent=2)
    
    return filepath


def evaluate_against_opponent(
    player_agent: OpenAIEndpointAgent,
    opponent_name: str,
    opponent_agent: ChessAgent,
    num_games: int = 10,
    verbose: bool = False,
    base_url: str = None,
    api_key: str = None,
    max_retries: int = None,
    template_file: str = None,
    debug: bool = False
) -> EvaluationResults:
    """
    Evaluate player agent against a specific opponent.
    
    Args:
        player_agent: The player agent being evaluated (used for config only)
        opponent_name: Name of the opponent (for reporting)
        opponent_agent: The opponent agent
        num_games: Number of games to play (must be even)
        verbose: Whether to print game progress
        base_url: Base URL for creating new player agents per game
        api_key: API key for creating new player agents per game
        max_retries: Max retries for creating new player agents per game
        template_file: Template file for creating new player agents per game
        debug: Debug mode for creating new player agents per game
    
    Returns:
        EvaluationResults object with statistics
    """
    if num_games % 2 != 0:
        raise ValueError("num_games must be even (for equal white/black distribution)")
    
    print(f"\n{'='*70}")
    print(f"EVALUATING AGAINST: {opponent_name}")
    print(f"{'='*70}")
    print(f"Playing {num_games} games ({num_games//2} as white, {num_games//2} as black)")
    
    # Create timestamp for this evaluation session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    games_per_color = num_games // 2
    
    def play_and_analyze_game(game_num: int, player_color: str):
        """Play a single game and analyze it."""
        # Create a fresh player agent for this game
        game_player_agent = OpenAIEndpointAgent(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            template_file=template_file,
            debug=debug
        )
        
        # Create a fresh opponent agent for this game (especially important for Stockfish)
        if isinstance(opponent_agent, StockfishAgent):
            game_opponent_agent = StockfishAgent(
                depth=opponent_agent.depth if hasattr(opponent_agent, 'depth') else 1,
                skill_level=opponent_agent.skill_level if hasattr(opponent_agent, 'skill_level') else 0
            )
        elif isinstance(opponent_agent, RandomAgent):
            game_opponent_agent = RandomAgent()
        else:
            game_opponent_agent = opponent_agent  # Fallback to shared instance
        
        try:
            game_result = play_game(game_player_agent, game_opponent_agent, player_color, game_num, verbose)
            
            # Analyze ACPL - create a new analyzer for each game since it closes after analyze_game
            white_acpl = 0.0
            black_acpl = 0.0
            if len(game_result["move_history"]) > 0:
                try:
                    analyzer = _StockfishAnalyzer(depth=20, movetime_ms=1000)
                    acpl_result = analyzer.analyze_game(game_result["move_history"])
                    white_acpl = acpl_result["white_acpl"]
                    black_acpl = acpl_result["black_acpl"]
                except Exception as e:
                    print(f"Warning: ACPL analysis failed for game {game_num}: {e}")
            
            stats = GameStats(
                result=game_result["result"],
                moves_played=game_result["moves_played"],
                white_time=game_result["white_time"],
                black_time=game_result["black_time"],
                white_acpl=white_acpl,
                black_acpl=black_acpl,
                player_color=player_color
            )
            
            # Save game log
            log_path = save_game_log(
                game_num=game_num,
                opponent_name=opponent_name,
                player_color=player_color,
                game_result=game_result,
                white_acpl=white_acpl,
                black_acpl=black_acpl,
                timestamp=timestamp
            )
            
            player_acpl = white_acpl if player_color == "white" else black_acpl
            player_time = game_result['white_time'] if player_color == "white" else game_result['black_time']
            
            print(f"✓ Game {game_num}/{num_games} ({player_color}): {game_result['result']} "
                  f"in {game_result['moves_played']} moves "
                  f"(Player ACPL: {player_acpl:.1f}, Time: {player_time:.2f}s)")
            
            return stats
        finally:
            # Clean up the game-specific opponent agent
            if hasattr(game_opponent_agent, 'close'):
                game_opponent_agent.close()
    
    # Play all games in parallel
    game_stats = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all games
        futures = []
        
        # Games as white
        for i in range(games_per_color):
            game_num = i + 1
            future = executor.submit(play_and_analyze_game, game_num, "white")
            futures.append(future)
        
        # Games as black
        for i in range(games_per_color):
            game_num = games_per_color + i + 1
            future = executor.submit(play_and_analyze_game, game_num, "black")
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                stats = future.result()
                game_stats.append(stats)
            except Exception as e:
                print(f"Error in game execution: {e}")
                import traceback
                traceback.print_exc()
    
    # Calculate statistics
    wins = 0
    draws = 0
    losses = 0
    total_acpl = 0.0
    total_time = 0.0
    total_moves = 0
    
    for stats in game_stats:
        result = stats.result
        
        # Determine outcome from player's perspective
        if stats.player_color == "white":
            if "White wins" in result:
                wins += 1
            elif "Black wins" in result:
                losses += 1
            else:
                draws += 1
            total_acpl += stats.white_acpl
            total_time += stats.white_time * stats.moves_played / 2  # Approximate
        else:
            if "Black wins" in result:
                wins += 1
            elif "White wins" in result:
                losses += 1
            else:
                draws += 1
            total_acpl += stats.black_acpl
            total_time += stats.black_time * stats.moves_played / 2  # Approximate
        
        total_moves += stats.moves_played
    
    avg_acpl = total_acpl / len(game_stats) if game_stats else 0.0
    avg_time = total_time / total_moves if total_moves > 0 else 0.0
    
    return EvaluationResults(
        opponent_name=opponent_name,
        total_games=num_games,
        wins=wins,
        draws=draws,
        losses=losses,
        avg_acpl=avg_acpl,
        avg_time_per_move=avg_time,
        games=game_stats
    )


def print_results(results: List[EvaluationResults]):
    """Print evaluation results in a nice format."""
    print("\n" + "="*70)
    print(" "*20 + "EVALUATION RESULTS")
    print("="*70)
    
    for result in results:
        print(f"Player vs {result.opponent_name}:")
        print(f"  Games Played:    {result.total_games}")
        print(f"  Wins:            {result.wins} ({result.wins/result.total_games*100:.1f}%)")
        print(f"  Draws:           {result.draws} ({result.draws/result.total_games*100:.1f}%)")
        print(f"  Losses:          {result.losses} ({result.losses/result.total_games*100:.1f}%)")
        print(f"  Average ACPL:    {result.avg_acpl:.2f}")
        print(f"  Avg Time/Move:   {result.avg_time_per_move:.3f}s")
    
    # Overall statistics
    total_games = sum(r.total_games for r in results)
    total_wins = sum(r.wins for r in results)
    total_draws = sum(r.draws for r in results)
    total_losses = sum(r.losses for r in results)
    overall_acpl = sum(r.avg_acpl * r.total_games for r in results) / total_games if total_games > 0 else 0
    overall_time = sum(r.avg_time_per_move * r.total_games for r in results) / total_games if total_games > 0 else 0
    
    print(f"\n{'='*70}")
    print(" "*25 + "OVERALL")
    print(f"{'='*70}")
    print(f"  Total Games:     {total_games}")
    print(f"  Total Wins:      {total_wins} ({total_wins/total_games*100:.1f}%)")
    print(f"  Total Draws:     {total_draws} ({total_draws/total_games*100:.1f}%)")
    print(f"  Total Losses:    {total_losses} ({total_losses/total_games*100:.1f}%)")
    print(f"  Overall ACPL:    {overall_acpl:.2f}")
    print(f"  Overall Time:    {overall_time:.3f}s per move")
    print(f"{'='*70}\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate a chess agent against multiple opponents"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:5000/v1",
        help="Base URL of the OpenAI-compatible API endpoint"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="dummy",
        help="API key for the endpoint"
    )
    parser.add_argument(
        "--games-per-opponent",
        type=int,
        default=10,
        help="Number of games to play per opponent (must be even)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of retries for illegal moves before resigning"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed game progress"
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=1,
        help="Stockfish opponent depth (default: 1)"
    )
    parser.add_argument(
        "--stockfish-skill",
        type=int,
        default=0,
        help="Stockfish opponent skill level 0-20 (default: 0, lower is weaker)"
    )
    parser.add_argument(
        "--template-file",
        type=str,
        default='player_agents/llm_agent_prompt_template.jinja',
        help="Path to Jinja2 template file for prompt formatting"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print input prompts and output responses for debugging"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.games_per_opponent % 2 != 0:
        print("Error: --games-per-opponent must be even")
        sys.exit(1)
    
    print("="*70)
    print(" "*20 + "CHESS AGENT EVALUATION")
    print("="*70)
    print(f"Endpoint:            {args.endpoint}")
    print(f"Games per opponent:  {args.games_per_opponent}")
    print(f"Max retries:         {args.max_retries}")
    print(f"ACPL analysis:       Enabled")
    print(f"Stockfish depth:     {args.stockfish_depth}")
    print(f"Stockfish skill:     {args.stockfish_skill}")
    print(f"Template file:       {args.template_file if args.template_file else 'Default (built-in)'}")
    print(f"Debug mode:          {'Enabled' if args.debug else 'Disabled'}")
    
    # Show logs directory
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    print(f"Game logs directory: {logs_dir}")
    
    # Create opponents
    opponents = []
    
    # Random agent
    # try:
    #     random_agent = RandomAgent()
    #     opponents.append(("Random Agent", random_agent))
    #     print(f"✓ Created Random Agent opponent")
    # except Exception as e:
    #     print(f"✗ Failed to create Random Agent: {e}")
    
    # Stockfish agent
    try:
        stockfish_agent = StockfishAgent(depth=args.stockfish_depth, skill_level=args.stockfish_skill, time_limit_ms=100)
        opponents.append((f"Stockfish (depth {args.stockfish_depth}, skill {args.stockfish_skill})", stockfish_agent))
        print(f"✓ Created Stockfish opponent (depth {args.stockfish_depth}, skill level {args.stockfish_skill})")
    except Exception as e:
        print(f"✗ Failed to create Stockfish agent: {e}")
        print("   Make sure Stockfish is installed on your system")
    
    if not opponents:
        print("\nError: No opponents available. Exiting.")
        sys.exit(1)
    
    # Run evaluation against each opponent using ThreadPoolExecutor
    def evaluate_opponent_task(opponent_name, opponent_agent):
        """Task wrapper for parallel evaluation."""
        # Create a dummy player agent just for config reference
        dummy_player_agent = OpenAIEndpointAgent(
            base_url=args.endpoint,
            api_key=args.api_key,
            max_retries=args.max_retries,
            template_file=args.template_file,
            debug=args.debug
        )
        
        try:
            result = evaluate_against_opponent(
                player_agent=dummy_player_agent,
                opponent_name=opponent_name,
                opponent_agent=opponent_agent,
                num_games=args.games_per_opponent,
                verbose=args.verbose,
                base_url=args.endpoint,
                api_key=args.api_key,
                max_retries=args.max_retries,
                template_file=args.template_file,
                debug=args.debug
            )
            return result
        except Exception as e:
            print(f"\nError evaluating against {opponent_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Clean up Stockfish agents (the original one, game-specific ones are cleaned up in evaluate_against_opponent)
            if hasattr(opponent_agent, 'close'):
                opponent_agent.close()
    
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all evaluation tasks
        future_to_opponent = {
            executor.submit(evaluate_opponent_task, opponent_name, opponent_agent): opponent_name
            for opponent_name, opponent_agent in opponents
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_opponent):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # Print final results
    if results:
        print_results(results)
    else:
        print("\nNo results to display.")


if __name__ == "__main__":
    main()