import json
import heapq
import numpy as np
import matplotlib.pyplot as plt

num_chunks = 2  # for 25 qubits with 5-qubit subcircuits => N is 25 in our case
num_bits = 10

def find_top_k_results(sub_probs, k=15):
    """
    Reconstructs the top K results. 
    Note: sub_probs should be passed in the correct order 
    (from MSB chunks to LSB chunks).
    """
    current_top = [(p, bs) for bs, p in sub_probs[0].items()]
    
    for i in range(1, len(sub_probs)):
        next_top = []
        for p_accum, bs_accum in current_top:
            for bs_new, p_new in sub_probs[i].items():
                combined_p = p_accum * p_new
                combined_bs = bs_accum + bs_new
                
                if len(next_top) < k:
                    heapq.heappush(next_top, (combined_p, combined_bs))
                else:
                    if combined_p > next_top[0][0]:
                        heapq.heappushpop(next_top, (combined_p, combined_bs))
        current_top = next_top
    
    return sorted(current_top, key=lambda x: x[0], reverse=True)

def get_target_stats(sub_probs, target_bs, baseline):
    """Calculates metrics for the target string using the provided sub_probs order."""
    chunks = [target_bs[i:i+5] for i in range(0, num_bits, 5)]
    chunk_probs = []
    total_p = 1.0
    
    for i, chunk in enumerate(chunks):
        p = sub_probs[i].get(chunk, 0.0)
        chunk_probs.append(p)
        total_p *= p
    
    gain = total_p / baseline if baseline > 0 else 0
    return {"total_p": total_p, "chunk_probs": chunk_probs, "gain": gain}

def plot_reconstruction(stats, top_15, target_bs, baseline):
    """Visualizes the fidelity per chunk and the global ranking."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    chunk_labels = [f"Chunk {i+1}\n({target_bs[i*5:i*5+5]})" for i in range(num_chunks)]
    colors = ['#2ecc71' if p > (1/32) else '#e74c3c' for p in stats['chunk_probs']]
    ax1.bar(chunk_labels, stats['chunk_probs'], color=colors, edgecolor='black', alpha=0.8)
    ax1.axhline(1/32, color='blue', linestyle='--', label='Uniform (1/32)')
    ax1.set_title("Subcircuit Probabilities (Corrected Order)", fontweight='bold')
    ax1.set_ylabel("Probability")
    ax1.legend()

    t15_probs = [x[0] for x in top_15]
    t15_labels = [f"Rank {i+1}" for i in range(len(top_15))]
    bar_colors = ['gold' if x[1] == target_bs else 'lightgray' for x in top_15]
    
    ax2.bar(t15_labels, t15_probs, color=bar_colors, edgecolor='black')
    ax2.axhline(baseline, color='red', linestyle=':', label='Baseline')
    ax2.set_yscale('log')
    ax2.set_title("Top 15 Global Bitstrings (Log Scale)", fontweight='bold')
    ax2.set_ylabel("Prob")
    plt.xticks(rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def extended_reconstruction(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    target_bs = data['input_bitstring']
    raw_subs = data['subcircuit_results']
    ordered_subs = raw_subs[::-1] 
    
    sub_probs = []
    json_expected_bs = ""
    for res in ordered_subs:
        s = res['shots']
        sub_probs.append({str(k): v/s for k, v in res['counts'].items()})
        json_expected_bs += res['evaluation']['expected_bitstring']
    
    baseline = 1 / (2**num_bits)
    
    stats = get_target_stats(sub_probs, target_bs, baseline)
    expected_stats = get_target_stats(sub_probs, json_expected_bs, baseline)
    
    top_15 = find_top_k_results(sub_probs, k=15)
    
    print("\n" + "="*50)
    print(f"REPORT FOR TARGET: {target_bs}")
    print(f"JSON GROUND TRUTH: {json_expected_bs}")
    print("="*50)
    print(f"Target Total Prob:   {stats['total_p']:.4e}")
    print(f"Truth Total Prob:    {expected_stats['total_p']:.4e}")
    print(f"Multiplicative Gain: {stats['gain']:.2f}x")
    print("-" * 50)
    
    plot_reconstruction(stats, top_15, target_bs, baseline)

if __name__ == "__main__":
    # Insert the results file from the RL agent here and make sure it is N bits
    FILE = 'qi_cut_qft_10q_Tuna-9_2parts_20260125_121843.json'
    
    extended_reconstruction(FILE)