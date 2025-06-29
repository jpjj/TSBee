from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def load_samples(file_a: str, file_b: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load samples from files. Replace this with your actual data loading method.
    """
    # Example: If your data is in CSV files with a single column
    # return np.loadtxt(file_a), np.loadtxt(file_b)
    algorithm_a = np.loadtxt(file_a)
    algorithm_b = np.loadtxt(file_b)

    return algorithm_a, algorithm_b


def descriptive_statistics(sample_a: np.ndarray, sample_b: np.ndarray) -> pd.DataFrame:
    """
    Compute descriptive statistics for both samples
    """
    stats_dict = {
        "Algorithm A": {
            "Mean": np.mean(sample_a),
            "Median": np.median(sample_a),
            "Std Dev": np.std(sample_a),
            "Min": np.min(sample_a),
            "Max": np.max(sample_a),
            "Size": len(sample_a),
        },
        "Algorithm B": {
            "Mean": np.mean(sample_b),
            "Median": np.median(sample_b),
            "Std Dev": np.std(sample_b),
            "Min": np.min(sample_b),
            "Max": np.max(sample_b),
            "Size": len(sample_b),
        },
    }

    return pd.DataFrame(stats_dict)


def visualize_distributions(sample_a: np.ndarray, sample_b: np.ndarray) -> None:
    """
    Create visualizations to compare the two distributions
    """
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(sample_a, kde=True, label="Algorithm A", alpha=0.6)
    sns.histplot(sample_b, kde=True, label="Algorithm B", alpha=0.6)
    plt.title("Distribution Comparison")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    # Box plot
    plt.subplot(1, 2, 2)
    data = pd.DataFrame({"Algorithm A": sample_a, "Algorithm B": sample_b})
    sns.boxplot(data=data)
    plt.title("Box Plot Comparison")

    plt.tight_layout()
    plt.savefig("distribution_comparison.png")
    plt.show()


def perform_t_test(sample_a: np.ndarray, sample_b: np.ndarray) -> Tuple[float, float]:
    """
    Perform an independent two-sample t-test
    Returns the t-statistic and p-value
    """
    t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    return t_stat, p_value


def perform_mann_whitney(sample_a: np.ndarray, sample_b: np.ndarray) -> Tuple[float, float]:
    """
    Perform a Mann-Whitney U test (non-parametric)
    Returns the U-statistic and p-value
    """
    u_stat, p_value = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
    return u_stat, p_value


def calculate_effect_size(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size
    """
    mean_a = np.mean(sample_a)
    mean_b = np.mean(sample_b)
    std_a = np.std(sample_a)
    std_b = np.std(sample_b)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((len(sample_a) - 1) * std_a**2 + (len(sample_b) - 1) * std_b**2)
        / (len(sample_a) + len(sample_b) - 2)
    )

    # Cohen's d
    cohen_d = (mean_b - mean_a) / pooled_std

    return cohen_d


def bootstrap_confidence_interval(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for the difference in means
    """
    diffs = []
    np.random.seed(42)

    for _ in range(n_bootstrap):
        bootstrap_a = np.random.choice(sample_a, size=len(sample_a), replace=True)
        bootstrap_b = np.random.choice(sample_b, size=len(sample_b), replace=True)

        diff = np.mean(bootstrap_b) - np.mean(bootstrap_a)
        diffs.append(diff)

    # Calculate confidence interval
    lower_bound = np.percentile(diffs, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(diffs, (1 + confidence) / 2 * 100)

    return lower_bound, upper_bound


def ab_test_report(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    alpha: float = 0.05,
    bootstrap_iterations: int = 10000,
) -> None:
    """
    Generate a complete A/B test report
    """
    print("=" * 50)
    print("A/B TEST ANALYSIS REPORT")
    print("=" * 50)

    # Descriptive statistics
    print("\n1. Descriptive Statistics:")
    stats_df = descriptive_statistics(sample_a, sample_b)
    print(stats_df)

    # Mean difference
    mean_diff = np.mean(sample_b) - np.mean(sample_a)
    mean_diff_pct = (mean_diff / np.mean(sample_a)) * 100
    print(f"\nRaw mean difference (B - A): {mean_diff:.4f}")
    print(f"Percentage difference: {mean_diff_pct:.2f}%")

    # T-test
    t_stat, p_value_t = perform_t_test(sample_a, sample_b)
    print("\n2. Two-Sample T-Test (Welch's t-test):")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value_t:.8f}")
    print(f"Statistically significant at α = {alpha}: {'Yes' if p_value_t < alpha else 'No'}")

    # Mann-Whitney U test
    u_stat, p_value_mw = perform_mann_whitney(sample_a, sample_b)
    print("\n3. Mann-Whitney U Test (Non-parametric):")
    print(f"U-statistic: {u_stat:.4f}")
    print(f"p-value: {p_value_mw:.8f}")
    print(f"Statistically significant at α = {alpha}: {'Yes' if p_value_mw < alpha else 'No'}")

    # Effect size
    cohen_d = calculate_effect_size(sample_a, sample_b)
    print("\n4. Effect Size:")
    print(f"Cohen's d: {cohen_d:.4f}")

    # Interpret effect size
    if abs(cohen_d) < 0.2:
        effect_interpretation = "negligible"
    elif abs(cohen_d) < 0.5:
        effect_interpretation = "small"
    elif abs(cohen_d) < 0.8:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    print(f"Effect size interpretation: {effect_interpretation}")

    # Bootstrap confidence interval
    lower_bound, upper_bound = bootstrap_confidence_interval(
        sample_a, sample_b, n_bootstrap=bootstrap_iterations
    )

    print("\n5. Bootstrap 95% Confidence Interval for Difference in Means (B - A):")
    print(f"Lower bound: {lower_bound:.4f}")
    print(f"Upper bound: {upper_bound:.4f}")
    print(f"Includes zero: {'Yes' if lower_bound <= 0 <= upper_bound else 'No'}")

    # Overall conclusion
    print("\n6. Overall Conclusion:")
    if p_value_t < alpha and p_value_mw < alpha:
        if mean_diff > 0:
            conclusion = "Algorithm B performs significantly better than Algorithm A."
        else:
            conclusion = "Algorithm A performs significantly better than Algorithm B."
    else:
        conclusion = "There is no statistically significant difference between the two algorithms."

    print(conclusion)

    if abs(cohen_d) < 0.2 and p_value_t < alpha:
        print(
            "Note: While the difference is statistically significant, the effect size is negligible,"
        )
        print("      suggesting the practical impact of this difference may be minimal.")

    print("\n7. Recommendation:")
    if mean_diff > 0 and p_value_t < alpha and abs(cohen_d) >= 0.2:
        recommendation = "Adopt Algorithm B as it shows better performance."
    elif mean_diff < 0 and p_value_t < alpha and abs(cohen_d) >= 0.2:
        recommendation = "Adopt Algorithm A as it shows better performance."
    elif p_value_t < alpha and abs(cohen_d) < 0.2:
        recommendation = "Either algorithm can be used as the difference, while statistically significant, has minimal practical impact."
    else:
        recommendation = "Either algorithm can be used as they perform similarly."

    print(recommendation)

    # Save descriptive statistics to CSV
    stats_df.to_csv("ab_test_stats.csv")
    print("\nDescriptive statistics saved to 'ab_test_stats.csv'")

    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_distributions(sample_a, sample_b)
    print("Visualizations saved to 'distribution_comparison.png'")

    print("\n" + "=" * 50)


def main():
    """
    Main function to run the A/B test analysis
    """
    # Replace these file paths with your actual data files
    file_a = "dontlookbits100k.csv"
    file_b = "nodontlookbits100k.csv"

    # Load data (or use synthetic data for demonstration)
    sample_a, sample_b = load_samples(file_a, file_b)

    # Run the complete analysis
    ab_test_report(sample_a, sample_b)


if __name__ == "__main__":
    main()
