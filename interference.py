import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
from transformers import GPT2Tokenizer

# Configuration
CONFIG = {
    "vocab_size": 50257,
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_gen_layers": 2,
    "num_context_layers": 1,
    "num_antennas": 16,
    "num_pyramid_levels": 8,
    "pyramid_pooling_factor": 2,
    "initial_interference_beta": 0.01,
    "max_interference_beta": 0.2,
    
    # Wave interference parameters
    "wave_propagation_speed": 1.0,
    "distance_decay_factor": 0.1,
    "max_frequency": 5.0,
    
    # Resonance parameters for long context
    "chunk_size": 384,
    "overlap_size": 64,
    "num_resonance_modes": 4,
    
    # Training parameters
    "batch_size": 8,
    "seq_length": 384,
    "learning_rate": 5e-4,
    "num_epochs": 20,
    "max_grad_norm": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Progressive training
    "warmup_epochs": 0,
    "debug_mode": True
}

class WaveInterferenceLayer(nn.Module):
    def __init__(self, hidden_dim, num_antennas, max_seq_len, config):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_antennas = num_antennas
        self.config = config
        
        # Fixed antenna positions distributed across sequence length
        antenna_positions = torch.linspace(0, max_seq_len-1, num_antennas)
        self.register_buffer('antenna_positions', antenna_positions)
        
        # Learnable antenna characteristics
        self.antenna_sensitivity = nn.Parameter(torch.ones(num_antennas) * 0.5)
        self.antenna_phase_shift = nn.Parameter(torch.zeros(num_antennas))
        
        # Signal generation networks
        self.signal_amplitude_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        self.signal_frequency_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.signal_phase_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        # Output projection
        self.interference_projection = nn.Linear(1, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Generate signal properties for each token
        signal_amplitudes = self.signal_amplitude_net(hidden_states)  # [B, S, 1]
        signal_frequencies = self.signal_frequency_net(hidden_states) * self.config["max_frequency"]  # [B, S, 1]
        signal_phases = self.signal_phase_net(hidden_states) * math.pi  # [B, S, 1]
        
        # Token positions
        token_positions = torch.arange(seq_len, device=hidden_states.device, dtype=torch.float32)
        
        # Calculate distances from each token to each antenna
        distances = torch.abs(
            token_positions.view(1, -1, 1) - 
            self.antenna_positions.view(1, 1, -1)
        )  # [1, S, A]
        
        # Wave propagation: phase accumulates with distance
        # Expand signal_frequencies to match distances: [B, S, 1] -> [B, S, A]
        propagation_phases = (
            signal_frequencies.expand(-1, -1, self.num_antennas) * 
            distances.expand(batch_size, -1, -1) / 
            self.config["wave_propagation_speed"]
        )  # [B, S, A]
        
        # Total phase at each antenna from each token
        total_phases = (
            signal_phases.expand(-1, -1, self.num_antennas) +  # [B, S, A]
            propagation_phases +  # [B, S, A]
            self.antenna_phase_shift.view(1, 1, -1).expand(batch_size, seq_len, -1)  # [B, S, A]
        )  # [B, S, A]
        
        # Signal amplitude at each antenna (with distance decay)
        received_amplitudes = (
            signal_amplitudes.expand(-1, -1, self.num_antennas) *  # [B, S, A]
            self.antenna_sensitivity.view(1, 1, -1).expand(batch_size, seq_len, -1) /  # [B, S, A]
            (1 + self.config["distance_decay_factor"] * distances.expand(batch_size, -1, -1))  # [B, S, A]
        )  # [B, S, A]
        
        # Wave interference: sum all signals at each antenna
        antenna_signals = torch.sum(
            received_amplitudes * torch.cos(total_phases), 
            dim=1
        )  # [B, A]
        
        # Vectorized reverse propagation: from antennas back to all token positions
        # Calculate all reverse distances at once
        reverse_distances = torch.abs(
            token_positions.view(1, -1, 1) - 
            self.antenna_positions.view(1, 1, -1)
        ).expand(batch_size, -1, -1)  # [B, S, A]
        
        # Reverse phase calculation for all positions
        reverse_phases = (
            signal_frequencies.expand(-1, -1, self.num_antennas) * 
            reverse_distances / self.config["wave_propagation_speed"] +
            self.antenna_phase_shift.view(1, 1, -1).expand(batch_size, seq_len, -1)
        )  # [B, S, A]
        
        # Reverse amplitude calculation
        reverse_amplitudes = (
            self.antenna_sensitivity.view(1, 1, -1).expand(batch_size, seq_len, -1) / 
            (1 + self.config["distance_decay_factor"] * reverse_distances)
        )  # [B, S, A]
        
        # Interference at each token from all antennas
        interference_signals = torch.sum(
            antenna_signals.unsqueeze(1).expand(-1, seq_len, -1) *  # [B, S, A]
            reverse_amplitudes * 
            torch.cos(reverse_phases),
            dim=-1, keepdim=True
        )  # [B, S, 1]
        
        # Project interference signal to hidden dimension
        interference_modulation = self.interference_projection(interference_signals)  # [B, S, H]
        
        # Apply interference with residual connection and normalization
        modulated_output = hidden_states + interference_modulation
        return self.layer_norm(modulated_output)

class TrueAbstractionPyramid(nn.Module):
    def __init__(self, input_dim, num_levels, pooling_factor):
        super().__init__()
        self.num_levels = num_levels
        self.pooling_factor = pooling_factor
        
        # Each level has its own processing
        self.level_lstms = nn.ModuleList()
        self.level_projections = nn.ModuleList()
        self.level_norms = nn.ModuleList()
        
        for i in range(num_levels):
            self.level_lstms.append(
                nn.LSTM(input_dim, input_dim, 1, batch_first=True, bidirectional=False)
            )
            self.level_projections.append(nn.Linear(input_dim, input_dim))
            self.level_norms.append(nn.LayerNorm(input_dim))
        
        # Cross-level attention
        self.cross_level_attention = nn.ModuleList()
        for i in range(num_levels - 1):
            self.cross_level_attention.append(
                nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
            )
        
        self.final_projection = nn.Linear(input_dim, input_dim)
        self.final_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        # Debug: Check input shape
        if x.dim() != 3:
            print(f"ERROR: TrueAbstractionPyramid received tensor with shape {x.shape}, expected 3D tensor")
            if x.dim() > 3:
                # Flatten extra dimensions
                x = x.view(x.size(0), x.size(1), -1)
            else:
                raise ValueError(f"Input tensor has {x.dim()} dimensions, expected 3")
        
        batch_size, seq_len, hidden_dim = x.shape
        
        # Process each abstraction level
        level_outputs = []
        current_input = x
        
        for level in range(self.num_levels):
            lstm_out, _ = self.level_lstms[level](current_input)
            level_out = self.level_projections[level](lstm_out)
            level_out = self.level_norms[level](level_out + current_input)
            level_outputs.append(level_out)
            
            # Create input for next level (higher abstraction)
            if level < self.num_levels - 1:
                new_seq_len = max(1, current_input.size(1) // self.pooling_factor)
                if new_seq_len < current_input.size(1):
                    current_input = F.adaptive_avg_pool1d(
                        level_out.transpose(1, 2), new_seq_len
                    ).transpose(1, 2)
                else:
                    current_input = level_out
        
        # Cross-level attention: lower levels attend to higher abstractions
        enhanced_levels = [level_outputs[0]]
        
        for i in range(len(level_outputs) - 1):
            lower_level = level_outputs[i]
            higher_level = level_outputs[i + 1]
            
            # Upsample higher level to match lower level length
            if higher_level.size(1) != lower_level.size(1):
                higher_level_upsampled = F.interpolate(
                    higher_level.transpose(1, 2),
                    size=lower_level.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                higher_level_upsampled = higher_level
            
            # Cross attention: lower attends to higher
            attended, _ = self.cross_level_attention[i](
                lower_level, higher_level_upsampled, higher_level_upsampled
            )
            
            enhanced = lower_level + attended
            enhanced_levels.append(enhanced)
        
        # Use the most enhanced version
        output = enhanced_levels[0]
        output = self.final_projection(output)
        output = self.final_norm(output + x)
        
        return output

class TrueAbstractionPyramid(nn.Module):
    def __init__(self, input_dim, num_levels, pooling_factor):
        super().__init__()
        self.num_levels = num_levels
        self.pooling_factor = pooling_factor
        
        # Each level has its own processing
        self.level_lstms = nn.ModuleList()
        self.level_projections = nn.ModuleList()
        self.level_norms = nn.ModuleList()
        
        for i in range(num_levels):
            self.level_lstms.append(
                nn.LSTM(input_dim, input_dim, 1, batch_first=True, bidirectional=False)
            )
            self.level_projections.append(nn.Linear(input_dim, input_dim))
            self.level_norms.append(nn.LayerNorm(input_dim))
        
        # Cross-level attention
        self.cross_level_attention = nn.ModuleList()
        for i in range(num_levels - 1):
            self.cross_level_attention.append(
                nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
            )
        
        self.final_projection = nn.Linear(input_dim, input_dim)
        self.final_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Process each abstraction level
        level_outputs = []
        current_input = x
        
        for level in range(self.num_levels):
            lstm_out, _ = self.level_lstms[level](current_input)
            level_out = self.level_projections[level](lstm_out)
            level_out = self.level_norms[level](level_out + current_input)
            level_outputs.append(level_out)
            
            # Create input for next level (higher abstraction)
            if level < self.num_levels - 1:
                new_seq_len = max(1, current_input.size(1) // self.pooling_factor)
                if new_seq_len < current_input.size(1):
                    current_input = F.adaptive_avg_pool1d(
                        level_out.transpose(1, 2), new_seq_len
                    ).transpose(1, 2)
                else:
                    current_input = level_out
        
        # Cross-level attention: lower levels attend to higher abstractions
        enhanced_levels = [level_outputs[0]]
        
        for i in range(len(level_outputs) - 1):
            lower_level = level_outputs[i]
            higher_level = level_outputs[i + 1]
            
            # Upsample higher level to match lower level length
            if higher_level.size(1) != lower_level.size(1):
                higher_level_upsampled = F.interpolate(
                    higher_level.transpose(1, 2),
                    size=lower_level.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                higher_level_upsampled = higher_level
            
            # Cross attention: lower attends to higher
            attended, _ = self.cross_level_attention[i](
                lower_level, higher_level_upsampled, higher_level_upsampled
            )
            
            enhanced = lower_level + attended
            enhanced_levels.append(enhanced)
        
        # Use the most enhanced version
        output = enhanced_levels[0]
        output = self.final_projection(output)
        output = self.final_norm(output + x)
        
        return output

class ResonantWaveWeaver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Learnable interference strength
        self.interference_beta = nn.Parameter(torch.tensor(config["initial_interference_beta"]))
        
        # Core components
        self.embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        
        self.generator_lstm = nn.LSTM(
            config["embedding_dim"], 
            config["hidden_dim"], 
            config["num_gen_layers"], 
            batch_first=True,
            dropout=0.1 if config["num_gen_layers"] > 1 else 0
        )
        
        # Wave interference layer
        self.wave_interference = WaveInterferenceLayer(
            config["hidden_dim"], 
            config["num_antennas"], 
            config["seq_length"], 
            config
        )
        
        # Abstraction pyramid
        self.abstraction_pyramid = TrueAbstractionPyramid(
            config["hidden_dim"], 
            config["num_pyramid_levels"], 
            config["pyramid_pooling_factor"]
        )
        
        self.fc_out = nn.Linear(config["hidden_dim"], config["vocab_size"])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config["hidden_dim"])
        
        # Resonance parameters for long sequences
        self.chunk_size = config.get("chunk_size", 384)
        self.overlap_size = config.get("overlap_size", 64)
        self.num_resonance_modes = config.get("num_resonance_modes", 4)
        
        self.resonance_memory = nn.Parameter(
            torch.randn(1, self.num_resonance_modes, config["hidden_dim"]) * 0.02
        )
        self.phase_memory = nn.Parameter(
            torch.zeros(1, self.num_resonance_modes)
        )
        
        self.resonance_updater = nn.Linear(config["hidden_dim"], self.num_resonance_modes)
        self.phase_updater = nn.Linear(config["hidden_dim"], self.num_resonance_modes)
        
        # Training state tracking
        self.current_epoch = 0
        self.enable_interference = False

    def set_epoch(self, epoch):
        """Update training phase based on epoch"""
        self.current_epoch = epoch
        self.enable_interference = epoch >= self.config.get("warmup_epochs", 3)
        
        # Gradually increase interference strength
        if self.enable_interference:
            progress = (epoch - self.config.get("warmup_epochs", 3)) / max(1, self.config["num_epochs"] - self.config.get("warmup_epochs", 3))
            target_beta = self.config["initial_interference_beta"] + progress * (self.config["max_interference_beta"] - self.config["initial_interference_beta"])
            self.interference_beta.data = torch.tensor(target_beta)

    def _standard_forward(self, input_ids, debug=False):
        """Standard forward pass for sequences <= chunk_size"""
        embedded_sequence = self.embedding(input_ids)
        generator_output, _ = self.generator_lstm(embedded_sequence)
        
        if debug and self.config.get("debug_mode", False):
            print(f"Generator output variance: {generator_output.var():.6f}")
            print(f"Generator output mean: {generator_output.mean():.6f}")
        
        # Apply wave interference only if enabled
        if self.enable_interference and self.training:
            # Apply true wave interference
            current_beta = torch.clamp(self.interference_beta, 0, self.config["max_interference_beta"])
            
            # Get interference-modulated representation
            interfered_output = self.wave_interference(generator_output)
            
            # Blend with original output
            modulated_representation = (
                (1 - current_beta) * generator_output + 
                current_beta * interfered_output
            )
        else:
            # Skip interference during warmup or evaluation
            modulated_representation = generator_output
        
        # Layer normalization for stability
        modulated_representation = self.layer_norm(modulated_representation)
        
        # Abstraction pyramid
        abstracted_representation = self.abstraction_pyramid(modulated_representation)
        
        # Final output
        final_logits = self.fc_out(abstracted_representation)
        
        if debug and self.config.get("debug_mode", False):
            print(f"Final logit variance: {final_logits.var():.6f}")
            print(f"Max logit: {final_logits.max():.6f}")
            print(f"Min logit: {final_logits.min():.6f}")
        
        return final_logits

    def _process_resonant_chunk(self, chunk, resonance_memory, phase_memory):
        """Process a single chunk with resonance coupling"""
        embedded_sequence = self.embedding(chunk)
        generator_output, _ = self.generator_lstm(embedded_sequence)
        
        batch_size, chunk_len, hidden_dim = generator_output.shape
        
        if self.enable_interference and self.training:
            # Apply wave interference
            current_beta = torch.clamp(self.interference_beta, 0, self.config["max_interference_beta"])
            interfered_output = self.wave_interference(generator_output)
            
            # Resonance coupling
            chunk_summary = torch.mean(generator_output, dim=1)
            
            resonance_scores = torch.matmul(
                generator_output,
                resonance_memory.transpose(-1, -2)
            )
            
            phase_expanded = phase_memory.unsqueeze(1)
            resonant_waves = torch.cos(resonance_scores + phase_expanded)
            
            resonance_weights = F.softmax(resonance_scores, dim=-1)
            resonant_enhancement = torch.sum(
                resonance_weights.unsqueeze(-1) * resonance_memory.unsqueeze(1), 
                dim=2
            )
            
            # Combine wave interference and resonance
            combined_interference = interfered_output + 0.2 * resonant_enhancement
            
            modulated = (
                (1 - current_beta) * generator_output + 
                current_beta * combined_interference
            )
            
            # Update resonance state
            resonance_update_weights = torch.sigmoid(self.resonance_updater(chunk_summary))
            new_resonance = (
                0.8 * resonance_memory + 
                0.2 * resonance_update_weights.unsqueeze(-1) * chunk_summary.unsqueeze(1)
            )
            
            phase_delta = self.phase_updater(chunk_summary)
            new_phase = (phase_memory + 0.1 * phase_delta) % (2 * math.pi)
        else:
            modulated = generator_output
            new_resonance = resonance_memory
            new_phase = phase_memory
        
        # Layer normalization
        modulated = self.layer_norm(modulated)
        
        # Abstraction pyramid
        abstracted = self.abstraction_pyramid(modulated)
        
        output = self.fc_out(abstracted)
        return output, new_resonance, new_phase

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        if seq_len <= self.chunk_size:
            return self._standard_forward(input_ids)
        
        # Process in overlapping resonant chunks for long sequences
        all_outputs = []
        current_resonance = self.resonance_memory.repeat(batch_size, 1, 1)
        current_phase = self.phase_memory.repeat(batch_size, 1)
        
        step_size = self.chunk_size - self.overlap_size
        
        for start_idx in range(0, seq_len - self.overlap_size, step_size):
            end_idx = min(start_idx + self.chunk_size, seq_len)
            chunk = input_ids[:, start_idx:end_idx]
            
            chunk_output, new_resonance, new_phase = self._process_resonant_chunk(
                chunk, current_resonance, current_phase
            )
            
            if start_idx == 0:
                all_outputs.append(chunk_output)
            else:
                overlap_blend = torch.linspace(0, 1, self.overlap_size).to(chunk_output.device)
                blended_overlap = (
                    overlap_blend.view(1, -1, 1) * chunk_output[:, :self.overlap_size] +
                    (1 - overlap_blend.view(1, -1, 1)) * all_outputs[-1][:, -self.overlap_size:]
                )
                
                all_outputs[-1] = torch.cat([
                    all_outputs[-1][:, :-self.overlap_size],
                    blended_overlap
                ], dim=1)
                
                if chunk_output.size(1) > self.overlap_size:
                    all_outputs.append(chunk_output[:, self.overlap_size:])
            
            current_resonance = new_resonance
            current_phase = new_phase
        
        return torch.cat(all_outputs, dim=1)

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, seq_length, stride_ratio=0.5):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = int(seq_length * stride_ratio)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.tokens = tokenizer.encode(text)
        print(f"Loaded {len(self.tokens)} tokens from input.txt")
        
        self.num_sequences = max(1, (len(self.tokens) - seq_length) // self.stride + 1)
        
        overlap_percent = (1 - stride_ratio) * 100
        print(f"Using {overlap_percent:.0f}% overlap between sequences")
        print(f"Dataset will have {self.num_sequences} sequences")
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_length + 1
        
        if end_idx > len(self.tokens):
            sequence = self.tokens[start_idx:] + [self.tokenizer.pad_token_id] * (end_idx - len(self.tokens))
        else:
            sequence = self.tokens[start_idx:end_idx]
        
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids

def compute_masked_loss(outputs, targets, tokenizer):
    """Compute loss while ignoring padding tokens"""
    mask = (targets != tokenizer.pad_token_id).float()
    
    outputs_flat = outputs.view(-1, CONFIG["vocab_size"])
    targets_flat = targets.view(-1)
    
    loss = F.cross_entropy(outputs_flat, targets_flat, reduction='none')
    masked_loss = loss * mask.view(-1)
    
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=outputs.device)
    
    return masked_loss.sum() / mask.sum()

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, device='cuda'):
    model.eval()
    
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Top-k sampling to prevent repetition
            if temperature > 0.1:
                top_k = 50
                top_logits, top_indices = torch.topk(next_token_logits, top_k)
                
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits[top_indices] = top_logits
                
                probabilities = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
            else:
                next_token = torch.argmax(next_token_logits).unsqueeze(0)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0].cpu().numpy(), skip_special_tokens=True)
    return generated_text

def train_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = ResonantWaveWeaver(CONFIG).to(CONFIG["device"])
    
    dataset = TextDataset('input.txt', tokenizer, CONFIG["seq_length"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    
    model.train()
    step = 0
    
    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n--- Epoch {epoch+1}/{CONFIG['num_epochs']} ---")
        
        model.set_epoch(epoch)
        
        if epoch < CONFIG.get("warmup_epochs", 3):
            print("WARMUP PHASE: Training basic LSTM without wave interference")
        else:
            print(f"WAVE INTERFERENCE PHASE: Beta = {model.interference_beta.item():.4f}")
        
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids, target_ids = input_ids.to(CONFIG["device"]), target_ids.to(CONFIG["device"])
            
            model.train()
            optimizer.zero_grad()
            
            outputs = model(input_ids)
            loss = compute_masked_loss(outputs, target_ids, tokenizer)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            step += 1
            
            if step % 500 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"Step {step}, Avg Loss: {avg_loss:.4f}, Current Loss: {loss.item():.4f}")
                
                if step % 1000 == 0:
                    print(f"\n--- Generation at Step {step} ---")
                    
                    model.eval()
                    
                    test_prompts = [
                        "In a realm of endless twilight",
                        "The city was",
                        "Once upon a time"
                    ]
                    
                    for prompt in test_prompts:
                        for temp in [0.3, 0.8]:
                            generated = generate_text(
                                model, tokenizer, prompt, 
                                max_length=50, temperature=temp, device=CONFIG["device"]
                            )
                            print(f"Prompt: '{prompt}' (temp={temp})")
                            print(f"Generated: '{generated}'\n")
                    
                    model.train()
        
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': CONFIG,
            'loss': avg_epoch_loss
        }, f'waveweaver_interference_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    print("WaveWeaver with True Wave Interference - PHYSICS-INSPIRED VERSION")
    print(f"Device: {CONFIG['device']}")
    print(f"Antennas: {CONFIG['num_antennas']}")
    print(f"Wave propagation speed: {CONFIG['wave_propagation_speed']}")
    print(f"Max frequency: {CONFIG['max_frequency']}")
    print(f"Warmup epochs: {CONFIG.get('warmup_epochs', 3)}")
    
    train_model()
