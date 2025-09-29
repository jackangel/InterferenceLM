import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tiktoken
from tqdm import tqdm
import math
import torch.nn.functional as F
import os

# --- 1. Configuration ---
CONFIG = {
    "tokenizer_name": "p50k_base",
    "vocab_size": 50257,
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_gen_layers": 2,
    "seq_length": 512,
    # Abstraction Pyramid Config
    "num_pyramid_levels": 3,
    "pyramid_pooling_factor": 2,
    # WaveWeaver-Solo Specific Config
    "num_antennas": 8,
    "interference_beta": 0.1,
    # Training Hyperparameters
    "num_epochs": 10,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "generate_every_n_steps": 250,
}

DATA_FILE = "input.txt"
CHECKPOINT_FILE = "waveweaver_checkpoint_tiktoken.pth"


# --- 2. Tiktoken Wrapper ---
class TiktokenWrapper:
    def __init__(self, encoding_name):
        self.encoder = tiktoken.get_encoding(encoding_name)

    def encode(self, text, return_tensors=None):
        encoded_ids = self.encoder.encode(text)
        if return_tensors == 'pt':
            return torch.tensor(encoded_ids, dtype=torch.long).unsqueeze(0)
        return encoded_ids

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.squeeze(0).tolist()
        return self.encoder.decode(ids)

# --- 3. Abstraction Pyramid ---
# --- MODIFIED: Added a check to handle short sequences during generation ---
class AbstractionPyramid(nn.Module):
    def __init__(self, input_dim, num_levels, pooling_factor):
        super().__init__()
        self.num_levels = num_levels
        self.pooling_factor = pooling_factor
        
        self.pyramid_layers = nn.ModuleList()
        current_dim = input_dim
        for _ in range(num_levels):
            self.pyramid_layers.append(
                nn.LSTM(current_dim, current_dim, 1, batch_first=True, bidirectional=True)
            )
        
        self.output_projection = nn.Linear(input_dim * (num_levels + 1), input_dim)

    def forward(self, x):
        pyramid_outputs = [x]
        current_representation = x
        
        for i in range(self.num_levels):
            # --- FIX STARTS HERE ---
            # Check if the current sequence length is large enough for pooling.
            if current_representation.size(1) < self.pooling_factor:
                # If not, we can't create a more abstract level.
                # We will reuse the most abstract representation we have (current_representation)
                # for all remaining pyramid levels to ensure the final concatenation is valid.
                upsampled_out = F.interpolate(current_representation.transpose(1, 2), size=x.size(1), mode='linear', align_corners=False).transpose(1, 2)
                pyramid_outputs.append(upsampled_out)
                continue # Move to the next loop iteration
            # --- FIX ENDS HERE ---

            # If the sequence is long enough, proceed as normal.
            lstm_out, _ = self.pyramid_layers[i](current_representation)
            lstm_out = lstm_out[:, :, :current_representation.size(-1)] + lstm_out[:, :, current_representation.size(-1):]
            
            pooled_out = F.avg_pool1d(
                lstm_out.transpose(1, 2),
                kernel_size=self.pooling_factor,
                stride=self.pooling_factor
            ).transpose(1, 2)
            
            upsampled_out = F.interpolate(pooled_out.transpose(1, 2), size=x.size(1), mode='linear', align_corners=False).transpose(1, 2)
            pyramid_outputs.append(upsampled_out)
            current_representation = pooled_out
            
        concatenated_outputs = torch.cat(pyramid_outputs, dim=-1)
        projected_output = self.output_projection(concatenated_outputs)
        
        return projected_output

# --- 4. WaveWeaver Components (Unchanged) ---
class AntennaConfigurationNetwork(nn.Module):
    def __init__(self, input_dim, num_antennas, antenna_pos_dim):
        super().__init__()
        self.num_antennas, self.antenna_pos_dim = num_antennas, antenna_pos_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, num_antennas * (antenna_pos_dim + 2))
        )
    def forward(self, e_vec, n_vec):
        control_input = torch.cat([e_vec, n_vec], dim=-1)
        params_flat = self.net(control_input)
        params = params_flat.view(control_input.size(0), self.num_antennas, -1)
        positions, amplitudes, phases = torch.split(params, [self.antenna_pos_dim, 1, 1], dim=-1)
        amplitudes = torch.sigmoid(amplitudes)
        phases = torch.sigmoid(phases) * 2 * math.pi
        return positions, amplitudes, phases

# --- 5. The Self-Modulating Language Model Architecture (Unchanged) ---
class WaveWeaverSolo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])

        self.generator_lstm = nn.LSTM(
            config["embedding_dim"], config["hidden_dim"],
            config["num_gen_layers"], batch_first=True, dropout=0.2
        )
        
        self.projection_to_hidden_e = nn.Linear(config["embedding_dim"], config["hidden_dim"], bias=False)
        self.projection_to_hidden_n = nn.Linear(config["embedding_dim"], config["hidden_dim"], bias=False)

        self.abstraction_pyramid = AbstractionPyramid(
            input_dim=config["hidden_dim"],
            num_levels=config["num_pyramid_levels"],
            pooling_factor=config["pyramid_pooling_factor"]
        )

        self.fc_out = nn.Linear(config["hidden_dim"], config["vocab_size"])

        self.context_analyzer_lstm = nn.LSTM(
            config["embedding_dim"], config["hidden_dim"], 1, batch_first=True
        )
        self.control_vector_generator = nn.Linear(
            config["hidden_dim"], config["embedding_dim"] * 2
        )

        self.acn = AntennaConfigurationNetwork(
            input_dim=config["embedding_dim"], 
            num_antennas=config["num_antennas"], 
            antenna_pos_dim=config["hidden_dim"]
        )
        
    def forward(self, input_ids):
        embedded_sequence = self.embedding(input_ids)

        _, (context_summary, _) = self.context_analyzer_lstm(embedded_sequence)
        context_summary = context_summary.squeeze(0)
        
        control_vectors_flat = self.control_vector_generator(context_summary)
        e_vec, n_vec = torch.chunk(control_vectors_flat, 2, dim=-1)

        generator_output, _ = self.generator_lstm(embedded_sequence)
        
        positions, amplitudes, phases = self.acn(e_vec, n_vec)
        
        rnn_expanded = generator_output.unsqueeze(2)
        pos_expanded = positions.unsqueeze(1)
        distances = F.cosine_similarity(rnn_expanded, pos_expanded, dim=-1)

        amp_exp = amplitudes.squeeze(-1).unsqueeze(1)
        phs_exp = phases.squeeze(-1).unsqueeze(1)

        wave_values = amp_exp * torch.cos(distances * 5.0 + phs_exp)

        enhance_waves, cancel_waves = torch.chunk(wave_values, 2, dim=-1)
        enhance_scalar = enhance_waves.sum(dim=-1, keepdim=True)
        cancel_scalar = cancel_waves.sum(dim=-1, keepdim=True)

        enhance_hidden_dir = self.projection_to_hidden_e(e_vec).unsqueeze(1)
        cancel_hidden_dir = self.projection_to_hidden_n(n_vec).unsqueeze(1)

        interference_vector = (enhance_scalar * enhance_hidden_dir) - (cancel_scalar * cancel_hidden_dir)
        
        modulated_representation = generator_output + self.config["interference_beta"] * interference_vector

        abstract_representation = self.abstraction_pyramid(modulated_representation)

        final_logits = self.fc_out(abstract_representation)
        
        return final_logits

# --- 6. Dataset and Generation (Unchanged) ---
class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        with open(filepath, 'r', encoding='utf-8') as f:
            self.tokens = tokenizer.encode(f.read())
    def __len__(self):
        return len(self.tokens) - self.seq_length
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_length + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

def generate_text(model, tokenizer, prompt, num_tokens_to_generate=500, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    for _ in range(num_tokens_to_generate):
        with torch.no_grad():
            logits = model(input_ids)
        
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs.index_fill_(1, indices_to_remove, 0)
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)
        
        next_token_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    model.train()
    return tokenizer.decode(input_ids)

# --- 7. Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = TiktokenWrapper(CONFIG["tokenizer_name"])
    
    model = WaveWeaverSolo(CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0
    global_step = 0

    if os.path.exists(CHECKPOINT_FILE):
        print(f"--- Found checkpoint: {CHECKPOINT_FILE} ---")
        print("--- Loading model and entering chat mode ---")
        
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("\nModel loaded. Type 'exit' or 'quit' to end the session.")
        while True:
            try:
                prompt = input("You: ")
                if prompt.lower() in ['exit', 'quit']:
                    break
                
                generated = generate_text(model, tokenizer, prompt, num_tokens_to_generate=500, top_p=0.9)
                print(f"Bot: {generated}")
            except KeyboardInterrupt:
                print("\nExiting chat mode.")
                break
            
    else:
        print("--- No checkpoint found, starting training from scratch ---")
        
        if not os.path.exists(DATA_FILE):
            print(f"Data file '{DATA_FILE}' not found. Creating a dummy file.")
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                f.write("This is a sample text file for the model to learn from. "
                        "Replace it with your own dataset for better results.")

        dataset = TextDataset(DATA_FILE, tokenizer, CONFIG["seq_length"])
        dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
        
        print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        print("--- Starting Training ---")
        
        for epoch in range(start_epoch, CONFIG["num_epochs"]):
            print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
            
            for input_ids, target_ids in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                
                optimizer.zero_grad()
                final_logits = model(input_ids)
                
                loss = criterion(
                    final_logits.view(-1, CONFIG['vocab_size']),
                    target_ids.view(-1)
                )
                
                loss.backward()
                optimizer.step()
                
                global_step += 1
                
                if global_step % CONFIG["generate_every_n_steps"] == 0:
                    print(f"\n--- Generation at Step {global_step} (Loss: {loss.item():.4f}) ---")
                    
                    prompts = ["In a realm of endless twilight", "The city was"]
                    for prompt in prompts:
                        generated = generate_text(model, tokenizer, prompt)
                        print(f"Prompt: '{prompt}'")
                        print(f"Generated: '{generated}'\n")
            
            print(f"--- End of Epoch {epoch+1}. Saving checkpoint... ---")
            checkpoint = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, CHECKPOINT_FILE)
            print(f"Checkpoint saved to {CHECKPOINT_FILE}")

        print("\n--- Training Finished ---")