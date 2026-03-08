import chess
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from chess_tournament import Player
from typing import Optional

class TransformerPlayer(Player):
    def __init__(self, name="Omer"):
        super().__init__(name)
        model_path = "omerK112345/chessColab1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.recent_moves = []

        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

    def evaluate_board(self, board, root_color):
        score = 0
        
        for piece_type in self.piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]

        for square in board.pieces(chess.PAWN, chess.WHITE):
            rank = chess.square_rank(square)
            score += rank * 0.5
            if rank == 6:
                score += 5

        for square in board.pieces(chess.PAWN, chess.BLACK):
            rank = 7 - chess.square_rank(square)
            score -= rank * 0.5
            if rank == 6:
                score -= 5

        return score if root_color == chess.WHITE else -score

    def minimax(self, board, depth, maximizing, root_color):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board, root_color)

        if maximizing:
            best = -9999
            for move in board.legal_moves:
                board.push(move)
                value = self.minimax(board, depth - 1, False, root_color)
                board.pop()
                best = max(best, value)
            return best
        else:
            best = 9999
            for move in board.legal_moves:
                board.push(move)
                value = self.minimax(board, depth - 1, True, root_color)
                board.pop()
                best = min(best, value)
            return best

    def model_score(self, fen, move):
        text = f"{fen} => {move}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        return -loss

    def get_move(self, fen):
        board = chess.Board(fen)
        root_color = board.turn 
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None

        best_move = None
        best_score = -9999

        for move in legal_moves:
            move_str = move.uci()
            
            if move_str in self.recent_moves and len(legal_moves) > 1:
                continue

            board.push(move)

            if board.is_checkmate():
                board.pop()
                self._update_memory(move_str)
                return move_str

            search_score = self.minimax(board, 2, False, root_color)
            transformer_score = self.model_score(fen, move_str)
            total_score = search_score + transformer_score

            if move.promotion:
                total_score += 9

            board.pop()

            if total_score > best_score:
                best_score = total_score
                best_move = move_str
                
        if best_move:
            self._update_memory(best_move)
            return best_move
        else:
            fallback = random.choice([m.uci() for m in legal_moves])
            self._update_memory(fallback)
            return fallback

    def _update_memory(self, move: str):
        self.recent_moves.append(move)
        if len(self.recent_moves) > 4:
            self.recent_moves.pop(0)
