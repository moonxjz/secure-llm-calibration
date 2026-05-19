#!/usr/bin/env python3
"""
Generate robustness evaluation set with adversarial mutations.
Creates variants of prompt injection attacks to test resilience.
"""

import random
import pandas as pd
from pathlib import Path

# Mutation strategies for adversarial robustness testing
MUTATION_STRATEGIES = {
    "rephrase": [
        # Rephrasing attacks to avoid exact pattern matching
        lambda x: x.replace("IGNORE ALL PREVIOUS", "forget all prior").replace("SYSTEM PROMPT", "system message"),
        lambda x: x.replace("Disregard", "Put aside").replace("Forget", "Dismiss"),
        lambda x: x.replace("INSTRUCTIONS", "directives").replace("OUTPUT", "result"),
        lambda x: x.replace("Tell me", "Reveal").replace("Show me", "Display"),
    ],
    "encoding": [
        # Unicode/whitespace variations
        lambda x: x.replace(" ", "\u00A0"),  # Non-breaking space
        lambda x: x.replace(" ", "\u2000"),  # En quad
        lambda x: "\n".join(x.split()),  # Excessive newlines
        lambda x: x.replace(".", ".\u200B"),  # Zero-width space
    ],
    "obfuscation": [
        # Instruction obfuscation
        lambda x: x.replace("SYSTEM", "SY" + "ST" + "EM").replace("PROMPT", "PRO" + "MPT"),
        lambda x: x.replace("ignore", "ign0re").replace("the", "th3"),  # L33t speak variants
        lambda x: x.upper() if random.random() > 0.5 else x.lower(),  # Case variation
        lambda x: " ".join([c for c in x if c.isalnum() or c.isspace()]),  # Remove special chars
    ],
    "context_shift": [
        # Add surrounding context to dilute attack signal
        lambda x: "Question: What is 2+2?\n\n" + x + "\n\nPlease respond concisely.",
        lambda x: "User query: " + x + "\n[End query]",
        lambda x: "Following is a customer service conversation:\n" + x,
        lambda x: x + "\n\nProvide a helpful response.",
    ],
    "fragmentation": [
        # Fragment instructions across multiple parts
        lambda x: x.replace("IGNORE ALL", "IGN\nORE A\nLL").replace("PREVIOUS", "PREV\nIOUS"),
        lambda x: " ... ".join(x.split()),  # Add separators
        lambda x: "\n".join([line.strip() for line in x.split()]) if len(x) > 20 else x,  # Newline fragmentation
    ]
}

def mutate_text(text: str, strategy: str, variant_num: int) -> str:
    """Apply mutation strategy to text."""
    if strategy not in MUTATION_STRATEGIES:
        return text
    
    mutations = MUTATION_STRATEGIES[strategy]
    mutation_func = mutations[variant_num % len(mutations)]
    
    try:
        mutated = mutation_func(text)
        # Verify mutation actually changed something
        return mutated if mutated != text else text
    except:
        return text  # Return original if mutation fails

def generate_robustness_set(base_dataset_path: str = "data/ATS_Customer_Support_500_balanced.csv",
                           output_path: str = "data/robustness_evaluation_200.csv"):
    """
    Generate robustness evaluation set from ATS Customer Support dataset.
    Creates original samples + 2-3 adversarial variants per attack.
    """
    print("\n=== Generating Robustness Evaluation Set ===")
    
    # Load base dataset
    try:
        df_base = pd.read_csv(base_dataset_path)
    except FileNotFoundError:
        print(f"✗ Base dataset not found: {base_dataset_path}")
        return None
    
    print(f"Loaded base dataset: {len(df_base)} rows")
    
    # Determine label column name
    label_col = None
    for col in ['is_attack', 'Answer', 'label']:
        if col in df_base.columns:
            label_col = col
            break
    
    if not label_col:
        print(f"✗ Cannot find label column in dataset. Columns: {df_base.columns.tolist()}")
        return None
    
    print(f"Using label column: '{label_col}'")
    
    # Extract attack samples
    attacks = df_base[df_base[label_col] == 1].reset_index(drop=True)
    if len(attacks) == 0:
        # Try 'attack' label
        attacks = df_base[df_base[label_col] == 'attack'].reset_index(drop=True)
    benign = df_base[df_base[label_col] == 0].reset_index(drop=True)
    if len(benign) == 0:
        benign = df_base[df_base[label_col] == 'benign'].reset_index(drop=True)
    
    print(f"Found: {len(attacks)} attacks, {len(benign)} benign samples")
    
    # Sample 50 attacks for mutation (to create ~100-150 total attack variants)
    sample_size = min(50, len(attacks))
    attacks_to_mutate = attacks.sample(n=sample_size, random_state=42)
    
    # Create robustness dataset with original + mutations
    robustness_rows = []
    strategies = list(MUTATION_STRATEGIES.keys())
    
    # Add original benign samples
    for idx, row in benign.head(100).iterrows():
        robustness_rows.append({
            'Question': row['Question'],
            'Answer': 0,
            'Scenario': row.get('Scenario', 'ATS_Customer_Support') if 'Scenario' in row else 'ATS_Customer_Support',
            'variant_type': 'original_benign',
            'variant_id': 0
        })
    
    # Add original + mutated attack samples
    for attack_idx, (idx, attack_row) in enumerate(attacks_to_mutate.iterrows()):
        original_text = attack_row['Question']
        
        # Add original attack
        robustness_rows.append({
            'Question': original_text,
            'Answer': 1,
            'Scenario': attack_row.get('Scenario', 'ATS_Customer_Support') if 'Scenario' in attack_row else 'ATS_Customer_Support',
            'variant_type': 'original_attack',
            'variant_id': 0
        })
        
        # Add 2-3 mutated variants per attack
        num_variants = random.randint(2, 3)
        for var_id in range(num_variants):
            strategy = strategies[var_id % len(strategies)]
            mutated_text = mutate_text(original_text, strategy, var_id)
            
            # Skip if mutation produced same result
            if mutated_text == original_text:
                continue
            
            robustness_rows.append({
                'Question': mutated_text,
                'Answer': 1,
                'Scenario': attack_row.get('Scenario', 'ATS_Customer_Support') if 'Scenario' in attack_row else 'ATS_Customer_Support',
                'variant_type': f'{strategy}_variant',
                'variant_id': var_id + 1
            })
    
    df_robustness = pd.DataFrame(robustness_rows)
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_robustness.to_csv(output_file, index=False)
    
    print(f"\n✓ Generated robustness evaluation set: {output_file}")
    print(f"  Total rows: {len(df_robustness)}")
    print(f"  - Original benign: {len([r for r in robustness_rows if r['variant_type'] == 'original_benign'])}")
    print(f"  - Original attacks: {len([r for r in robustness_rows if r['variant_type'] == 'original_attack'])}")
    print(f"  - Mutated variants: {len([r for r in robustness_rows if 'variant' in r['variant_type']])}")
    print(f"\n  Variant types distribution:")
    for vtype in set(r['variant_type'] for r in robustness_rows):
        count = len([r for r in robustness_rows if r['variant_type'] == vtype])
        print(f"    - {vtype}: {count}")
    
    return output_file

if __name__ == "__main__":
    generate_robustness_set()
