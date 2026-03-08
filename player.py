import chess
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from chess_tournament import Player
from typing import Optional

class TransformerPlayer(Player):
    def __init__(self, name: str = "Omer"):
        super().__init__(name)
        model_path = "omerK112345/chessColab1" 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.recent_moves = []
        self.piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves_str = [move.uci() for move in board.legal_moves]
        if not legal_moves_str: return None 
            
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                self._update_memory(move.uci())
                return move.uci()
            board.pop()
        
        prompt = f"{fen} => "
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, max_new_tokens=5, pad_token_id=self.tokenizer.eos_token_id,
                num_beams=5, num_return_sequences=5, early_stopping=True
            )
            
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            try: predicted_move = decoded.split("=>")[1].strip()[:5].strip()
            except IndexError: continue
                
            if predicted_move in legal_moves_str:
                if predicted_move in self.recent_moves and len(legal_moves_str) > 1:
                    continue
                    
                move_obj = chess.Move.from_uci(predicted_move)
                board.push(move_obj)
                opponent_can_mate = False
                
                if not board.is_checkmate():
                    for opp_move in board.legal_moves:
                        board.push(opp_move)
                        if board.is_checkmate(): opponent_can_mate = True
                        board.pop()
                        if opponent_can_mate: break
                            
                board.pop() 
                if not opponent_can_mate:
                    self._update_memory(predicted_move)
                    return predicted_move
                    
        best_fallback = None
        best_score = -1
        
        for move in board.legal_moves:
            move_str = move.uci()
            if move_str in self.recent_moves and len(legal_moves_str) > 1: continue 
                
            score = 0
            if board.is_capture(move):
                target = board.piece_at(move.to_square)
                if target: score += self.piece_values.get(target.piece_type, 0)
            if move.promotion == chess.QUEEN: score += 9
                
            if score >= best_score:
                best_score = score
                best_fallback = move_str
                
        if best_fallback and best_score > 0:
            self._update_memory(best_fallback)
            return best_fallback
        else:
            fallback = random.choice(legal_moves_str)
            self._update_memory(fallback)
            return fallback

    def _update_memory(self, move: str):
        self.recent_moves.append(move)
        if len(self.recent_moves) > 4:
            self.recent_moves.pop(0)
