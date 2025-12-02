import random
import argparse
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI API compatible endpoint for chess move generation"""
    try:
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
        
        # Parse the message to extract legal moves
        legal_moves = []
        if 'legal moves:' in user_message.lower():
            # Extract the legal moves portion
            parts = user_message.split('legal moves:')
            if len(parts) > 1:
                moves_part = parts[1].strip()
                # Split by spaces to get individual moves
                legal_moves = [m.strip() for m in moves_part.split() if m.strip()]
        
        if not legal_moves:
            return jsonify({"error": "No legal moves found in message"}), 400
        
        # Select a random legal move
        random_move = random.choice(legal_moves)
        
        # Format response in OpenAI API format
        response = {
            "id": f"chatcmpl-{random.randint(100000, 999999)}",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "random-chess-agent",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"<think>Selecting a random legal move from the available options</think><uci_move>{random_move}</uci_move>"
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
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Chess Agent Flask Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    args = parser.parse_args()
    
    print(f"Starting Random Chess Agent server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)