## Copy from https://github.com/AIcrowd/Global-Chess-Challenge-2025-Baselines
import chess

def encode_legal_moves(board):
    """Encode legal moves as special token sequence."""
    color = 'White' if board.turn == chess.WHITE else 'Black'
    promo_map = {chess.QUEEN: f'<{color}_Queen>', chess.ROOK: f'<{color}_Rook>', 
                 chess.BISHOP: f'<{color}_Bishop>', chess.KNIGHT: f'<{color}_Knight>'}
    
    moves = [f"<{chess.square_name(m.from_square)}><{chess.square_name(m.to_square)}>"
             + (promo_map[m.promotion] if m.promotion else "")
             for m in board.legal_moves]
    
    return " ".join(moves)

def piece_to_token(piece):
    if piece is None:
        return '<blank>'
    
    piece_map = {
        (chess.PAWN, chess.WHITE): '<White_Pawn>',
        (chess.KNIGHT, chess.WHITE): '<White_Knight>',
        (chess.BISHOP, chess.WHITE): '<White_Bishop>',
        (chess.ROOK, chess.WHITE): '<White_Rook>',
        (chess.QUEEN, chess.WHITE): '<White_Queen>',
        (chess.KING, chess.WHITE): '<White_King>',
        (chess.PAWN, chess.BLACK): '<Black_Pawn>',
        (chess.KNIGHT, chess.BLACK): '<Black_Knight>',
        (chess.BISHOP, chess.BLACK): '<Black_Bishop>',
        (chess.ROOK, chess.BLACK): '<Black_Rook>',
        (chess.QUEEN, chess.BLACK): '<Black_Queen>',
        (chess.KING, chess.BLACK): '<Black_King>',
    }
    
    return piece_map[(piece.piece_type, piece.color)]

def encode_board_position(fen):
    """Encode FEN to special token sequence."""
    board = chess.Board(fen)
    
    # Board tokens
    tokens = [f"<{chess.square_name(sq)}>{piece_to_token(board.piece_at(sq))}" 
              for sq in chess.SQUARES]
    
    # Metadata
    parts = fen.split()
    side = "White" if parts[1] == 'w' else "Black"
    other_info = f"{parts[2]}|{parts[3]}|{parts[4]}|{parts[5]}"
    legal_moves = encode_legal_moves(board)

    return "".join(tokens) + f"|{side}|{other_info}", legal_moves


def add_chess_tokens(model, tokenizer):
    # Generate all square tokens (a1 through h8)
    squares = [f"<{chess.square_name(sq)}>" for sq in chess.SQUARES]
    
    # Piece tokens
    pieces = [
        '<White_Pawn>', '<White_Knight>', '<White_Bishop>', 
        '<White_Rook>', '<White_Queen>', '<White_King>',
        '<Black_Pawn>', '<Black_Knight>', '<Black_Bishop>',
        '<Black_Rook>', '<Black_Queen>', '<Black_King>',
        '<blank>'
    ]
    
    special_tokens = {
        'additional_special_tokens': squares + pieces
    }
    
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))