import random
import argparse
from flask import Flask, request, jsonify
from stockfish import Stockfish

app = Flask(__name__)

# Initialize Stockfish (you may need to specify the path to stockfish binary)
# Common paths: /usr/games/stockfish, /usr/local/bin/stockfish, or just 'stockfish' if in PATH
try:
    stockfish = Stockfish(path="stockfish", depth=1, parameters={"Skill Level": 0})
except Exception as e:
    print(f"Warning: Could not initialize Stockfish with default path. Error: {e}")
    print("You may need to specify the correct path to the Stockfish binary.")
    stockfish = None

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI API compatible endpoint for chess move generation using Stockfish"""
    try:
        if stockfish is None:
            return jsonify({"error": "Stockfish engine not initialized"}), 500
            
        data = request.json
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break
        
        if not user_message:
            return jsonify({"error": "No user message found"}), 400
        
        # Parse the message to extract FEN position and legal moves
        fen_position = None
        legal_moves = []
        
        # Look for FEN position
        if 'fen:' in user_message.lower():
            parts = user_message.split('FEN:')
            if len(parts) > 1:
                fen_part = parts[1].strip().split('\n')[0].strip()
                fen_position = fen_part
        
        # Extract legal moves for fallback
        if 'legal moves:' in user_message.lower():
            parts = user_message.split('legal moves:')
            if len(parts) > 1:
                moves_part = parts[1].strip()
                legal_moves = [m.strip() for m in moves_part.split() if m.strip()]
        
        if not fen_position:
            return jsonify({"error": "No FEN position found in message"}), 400
        
        if not legal_moves:
            return jsonify({"error": "No legal moves found in message"}), 400
        
        # Set the position in Stockfish
        stockfish.set_fen_position(fen_position)
        
        # Get the best move from Stockfish at depth 1, skill level 0
        best_move = stockfish.get_best_move()
        
        # Validate that the move is in legal moves, otherwise fall back to random
        if best_move not in legal_moves:
            print(f"Warning: Stockfish suggested {best_move} which is not in legal moves. Falling back to random.")
            best_move = random.choice(legal_moves)
        
        # Format response in OpenAI API format
        response = {
            "id": f"chatcmpl-{random.randint(100000, 999999)}",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "stockfish-chess-agent",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"<think>Using Stockfish level 0 depth 1 to analyze position</think><uci_move>{best_move}</uci_move>"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if stockfish is None:
        return jsonify({"status": "unhealthy", "error": "Stockfish not initialized"}), 500
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stockfish Chess Agent Flask Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--stockfish-path', type=str, default='stockfish', help='Path to Stockfish binary')
    args = parser.parse_args()
    
    # Reinitialize Stockfish with custom path if provided
    if args.stockfish_path != 'stockfish':
        try:
            stockfish = Stockfish(path=args.stockfish_path, depth=1, parameters={"Skill Level": 0})
            print(f"Stockfish initialized with custom path: {args.stockfish_path}")
        except Exception as e:
            print(f"Error initializing Stockfish with path {args.stockfish_path}: {e}")
    
    print(f"Starting Stockfish Chess Agent server (Level 0, Depth 1) on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

