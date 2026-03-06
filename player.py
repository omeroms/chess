import chess
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from chess_tournament import Player
from typing import Optional

class TransformerPlayer(Player):
    def __init__(self, name: str = "GPT2-Student"):
        super().__init__(name)
        
        # Pulls your saved model directly from your Hugging Face account
        # UPDATED to your exact repository name!
        model_path = "omerK112345/chessColab1" 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = [move.uci() for move in board.legal_moves]
        
        if not legal_moves:
            return None 

        prompt = f"{fen} => "
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_new_tokens=5, 
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.3
            )
            
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            predicted_move = decoded.split("=>")[1].strip()[:5].strip()
        except IndexError:
            predicted_move = ""

        # Legal move filter
        if predicted_move in legal_moves:
            return predicted_move
        else:
            return random.choice(legal_moves)
