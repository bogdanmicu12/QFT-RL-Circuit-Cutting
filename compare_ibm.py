import json
import matplotlib.pyplot as plt
import numpy as np

def int_to_bitstring(val, width):
    return format(int(val), f'0{width}b')

def load_results(filename):
    with open(filename, "r") as f:
        return json.load(f)

full_data = load_results("qft_results_full_8.json")
cut_data = load_results("qft_results_cut_8.json")

full_probs_raw = full_data["circuit_0"]["probabilities"]
full_probs = {int_to_bitstring(k, 8): v for k, v in full_probs_raw.items()}

sub1_raw = cut_data["circuit_0"]["probabilities"]
sub2_raw = cut_data["circuit_1"]["probabilities"]

reconstructed_probs = {}
for k1, p1 in sub1_raw.items():
    for k2, p2 in sub2_raw.items():
        b1 = int_to_bitstring(k1, 4)
        b2 = int_to_bitstring(k2, 4)
        
        combined_bitstring = b2 + b1 
        reconstructed_probs[combined_bitstring] = p1 * p2

target = "00000000"
full_pos = full_probs.get(target, 0)
cut_pos = reconstructed_probs.get(target, 0)

all_keys = set(full_probs.keys()) | set(reconstructed_probs.keys())
fidelity = sum(np.sqrt(full_probs.get(k, 0) * reconstructed_probs.get(k, 0)) for k in all_keys)**2

print(f"--- QFT Comparison Metrics ---")
print(f"Target State ({target}) PoS:")
print(f"  Full: {full_pos:.4f}")
print(f"  Cut:  {cut_pos:.4f}")
print(f"Distribution Fidelity: {fidelity:.4f}")

top_n = 15
sorted_keys = sorted(full_probs.keys(), key=lambda x: full_probs[x], reverse=True)[:top_n]

full_vals = [full_probs.get(k, 0) for k in sorted_keys]
cut_vals = [reconstructed_probs.get(k, 0) for k in sorted_keys]

x = np.arange(len(sorted_keys))
plt.figure(figsize=(14, 7))
plt.bar(x - 0.2, full_vals, 0.4, label=f'Full Circuit (P_s= {full_pos:.2%})', alpha=0.8)
plt.bar(x + 0.2, cut_vals, 0.4, label=f'Cut Circuit Est (P_s= {cut_pos:.2%})', alpha=0.8)

plt.title(f"QFT 8-Qubit: Full vs. Cut Reconstruction (Fidelity: {fidelity:.2%})")
plt.xlabel("Output Bitstrings (Top 15)")
plt.ylabel("Probability")
plt.xticks(x, sorted_keys, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()