import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class MetricsAnalyzer:
    
    @staticmethod
    def generate_report(results, baselines):
        report = []
        
        if "phase2" in results:
            phase2 = results["phase2"]
            report.append(f"\nRL Agent (Phase 2) Results:")
            report.append(f"Average Fidelity: {phase2.get('avg_fidelity', 0):.4f}")
            report.append(f"Max Fidelity: {phase2.get('max_fidelity', 0):.4f}")
            report.append(f"Number of Trials: {len(phase2.get('trials', []))}")
        
        if baselines:
            report.append(f"\nBaseline Methods:")
            for name, baseline in baselines.items():
                report.append(f"  {name}:")
                report.append(f"Fidelity: {baseline.fidelity_estimate:.4f}")
                report.append(f"Overhead: {baseline.cutting_overhead:.1f}x")
                report.append(f"Num Cuts: {baseline.num_cuts}")
        
        return "\n".join(report)


class ResultsVisualizer:
    
    @staticmethod
    def plot_baseline_comparison(baselines, save_path = None, rl_results = None):
        if not baselines:
            return
        
        names = list(baselines.keys())
        fidelities = [b.fidelity_estimate for b in baselines.values()]
        widths = [b.max_partition_width for b in baselines.values()]
        overheads = [b.cutting_overhead for b in baselines.values()]
        num_cuts = [b.num_cuts for b in baselines.values()]
        
        rl_name = "RL Agent"
        if rl_results:
            if isinstance(rl_results, dict):
                rl_fidelity = rl_results.get("best_fidelity", rl_results.get("avg_fidelity", 0.9))
                rl_cuts = rl_results.get("num_cuts", 1)
                rl_overhead = 4.0 ** rl_cuts
                rl_width = rl_results.get("max_partition_width", max(widths) if widths else 4)
                
                names.insert(0, rl_name)
                fidelities.insert(0, rl_fidelity)
                widths.insert(0, rl_width)
                overheads.insert(0, rl_overhead)
                num_cuts.insert(0, rl_cuts)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        if rl_results:
            colors = [np.array([1.0, 0.0, 0.0, 1.0])] + list(colors[:-1])
        
        ax = axes[0, 0]
        bars = ax.bar(range(len(names)), fidelities, color=colors)
        ax.set_ylabel("Fidelity Estimate", fontsize=11)
        ax.set_title("Baseline vs RL Agent: Fidelity Comparison", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        
        ax.set_ylim([0, 1.10])
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            label_y = height + 0.02
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f"{height:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax = axes[0, 1]
        bars = ax.bar(range(len(names)), widths, color=colors)
        ax.set_ylabel("Max Partition Width", fontsize=11)
        ax.set_title("Partition Size Comparison", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, max(widths)*1.15]) 
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height+0.2,
                   f"{int(height)}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax = axes[1, 0]
        bars = ax.bar(range(len(names)), overheads, color=colors)
        ax.set_ylabel("Cutting Overhead", fontsize=11)
        ax.set_title("Classical Sampling Overhead", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        
        max_overhead = max(overheads)
        if max_overhead > 1:
            max_power = int(np.ceil(np.log(max_overhead) / np.log(4)))
            ticks_powers = range(0, max_power + 2)  
            ticks = [4**p for p in ticks_powers]
            ax.set_yscale("log")
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"$4^{p}$" for p in ticks_powers])
        else:
            ax.set_ylim([0, 2])
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 1:
                power = np.log(height) / np.log(4)
                label = f"$4^{{{power:.1f}}}$" if power != int(power) else f"$4^{int(power)}$"
            else:
                label = "1"
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.15,
                   label, ha="center", va="bottom", fontsize=10, fontweight="bold")
 
        ax = axes[1, 1]
        bars = ax.bar(range(len(names)), num_cuts, color=colors)
        ax.set_ylabel("Number of Cuts", fontsize=11)
        ax.set_title("Number of Wire Cuts Comparison", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        
        max_cuts = max(num_cuts) if num_cuts else 1
        ax.set_ylim([0, max_cuts * 1.15])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f"{int(height)}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


class ComprehensiveEvaluator:
    def __init__(self, output_dir = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, training_results, baseline_results,
                save_plots=True):
        
        analyzer = MetricsAnalyzer()
        report = analyzer.generate_report(training_results, baseline_results)
        print(report)
        
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        
        if save_plots:
            visualizer = ResultsVisualizer()

            rl_results = None
            if training_results and "phase2" in training_results:
                phase2 = training_results["phase2"]
                num_cuts = phase2.get("best_num_cuts", 1)
                num_qubits = training_results.get("num_qubits", 8)
                num_partitions = num_cuts + 1
                estimated_max_width = int(np.ceil(num_qubits / num_partitions))
                
                rl_results = {
                    "best_fidelity": phase2.get("max_fidelity", phase2.get("avg_fidelity", 0.9)),
                    "avg_fidelity": phase2.get("avg_fidelity", 0.9),
                    "num_cuts": num_cuts,
                    "max_partition_width": phase2.get("best_max_partition_width", estimated_max_width)
                }
            
            if baseline_results:
                visualizer.plot_baseline_comparison(
                    baseline_results,
                    save_path=self.output_dir / "baseline_comparison.png",
                    rl_results=rl_results
                )
            
            print(f"\nPlots saved to {self.output_dir}")
        
        return {
            "report": report,
            "report_path": str(report_path)
        }

