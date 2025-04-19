import os
import pandas as pd
import RNA
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def ensure_float(value):
    """Convert value to float if it's a string."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return value

def read_dbn_file(dbn_file):
    """Read RNA sequence and structure from a dbn file"""
    sequence = ""
    structure = ""
    
    with open(dbn_file, 'r') as f:
        lines = f.readlines()
        header_lines = 0
        for line in lines:
            if line.startswith('#'):
                header_lines += 1
            else:
                break
        content_lines = [line.strip() for line in lines[header_lines:]]
        
        if len(content_lines) >= 2:
            sequence = content_lines[0]
            structure = content_lines[1]
    
    return sequence, structure

def safe_log(value):
    """Safely compute logarithm, handling type conversions and edge cases."""
    try:
        float_val = ensure_float(value)
        if float_val <= 0:
            return 0.0
        return RNA.log(float_val)
    except Exception:
        return 0.0

def extract_global_thermodynamic_features(sequence):
    """
    Extract global thermodynamic features from RNA sequence using ViennaRNA
    
    Parameters:
    ----------
    sequence : str
        RNA sequence
        
    Returns:
    -------
    dict
        Dictionary containing thermodynamic features
    """
    features = {}
    
    try:
        # Create fold compound
        fc = RNA.fold_compound(sequence)
        
        # Calculate minimum free energy and structure
        (mfe_structure, mfe) = fc.mfe()
        features['minimum_free_energy'] = ensure_float(mfe)
        features['mfe_structure'] = mfe_structure
        
        # Calculate partition function and ensemble free energy
        (ensemble_free_energy, partition_function) = fc.pf()
        features['ensemble_free_energy'] = ensure_float(ensemble_free_energy)
        features['partition_function'] = ensure_float(partition_function)
        
        # Calculate difference between ensemble free energy and MFE
        e_free = ensure_float(features['ensemble_free_energy'])
        m_free = ensure_float(features['minimum_free_energy'])
        features['ensemble_diversity'] = e_free - m_free
        
        # Get base pair probabilities
        bpp = fc.bpp()
        
        # Calculate average base pair probability for pairs in the MFE structure
        pair_probs = []
        stack = []
        
        for i, char in enumerate(mfe_structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    # ViennaRNA uses 1-based indexing
                    if i+1 < len(bpp) and j+1 < len(bpp[0]):
                        # Ensure the probability is a float
                        prob = ensure_float(bpp[j+1][i+1])
                        pair_probs.append(prob)
        
        if pair_probs:
            features['avg_pair_probability'] = sum(pair_probs) / len(pair_probs)
        else:
            features['avg_pair_probability'] = 0.0
        
        # Calculate positional entropy
        positional_entropy = 0.0
        try:
            for i in range(1, len(sequence) + 1):
                for j in range(i + 1, len(sequence) + 1):
                    if i < len(bpp) and j < len(bpp[i]):
                        p = ensure_float(bpp[i][j])
                        if p > 0:
                            positional_entropy -= p * safe_log(p)
            
            features['positional_entropy'] = positional_entropy
        except Exception as e:
            print(f"Error calculating positional entropy: {e}")
            features['positional_entropy'] = 0.0
            
        # Additional features from RNAplfold for local pairing probabilities
        # This would need to be implemented separately with RNA.pfl_fold()
        
    except Exception as e:
        print(f"Error calculating thermodynamic features: {e}")
        features = {
            'mfe_structure': None,
            'minimum_free_energy': None,
            'ensemble_free_energy': None,
            'partition_function': None,
            'ensemble_diversity': None,
            'avg_pair_probability': None,
            'positional_entropy': None
        }
    
    return features

def process_file(dbn_file):
    """Process a single dbn file"""
    try:
        # Get RNA ID
        rna_id = os.path.basename(dbn_file).replace('.dbn', '')
        
        # Read dbn file - we only need the sequence
        sequence, _ = read_dbn_file(dbn_file)
        
        # Skip sequences that are too long or too short
        if len(sequence) < 10 or len(sequence) > 2000:
            return None
        
        # Extract global thermodynamic features using only sequence
        thermo_features = extract_global_thermodynamic_features(sequence)
        
        # Add RNA ID and sequence length
        thermo_features['rna_id'] = rna_id
        thermo_features['sequence_length'] = len(sequence)
        
        return thermo_features
    except Exception as e:
        print(f"Error processing {dbn_file}: {e}")
        return None

def process_all_dbn_files(directory, output_file, max_files=None, num_workers=4):
    """Process all dbn files in a directory using ProcessPoolExecutor instead of Pool"""
    # Get all dbn files
    print("Finding dbn files...")
    dbn_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.endswith('.dbn')]
    
    if max_files:
        dbn_files = dbn_files[:max_files]
    
    print(f"Found {len(dbn_files)} dbn files. Starting processing with {num_workers} workers.")
    
    # Process in smaller batches
    batch_size = 500
    all_features = []
    
    # Process in batches to show progress and avoid memory issues
    for i in range(0, len(dbn_files), batch_size):
        batch = dbn_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(dbn_files)-1)//batch_size + 1} ({len(batch)} files)")
        
        # Use ProcessPoolExecutor which works better on MacOS
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_file, file) for file in batch]
            
            # Show progress with simple counter
            completed = 0
            with tqdm(total=len(batch)) as pbar:
                for future in futures:
                    try:
                        result = future.result(timeout=60)  # Add timeout
                        if result is not None:
                            results.append(result)
                        completed += 1
                        pbar.update(1)
                        
                        # Save intermediate results periodically
                        if len(results) % 100 == 0 and results:
                            print(f"Processed {completed}/{len(batch)} files in current batch")
                    except Exception as e:
                        print(f"Error in worker process: {e}")
        
        all_features.extend(results)
        
        # Save intermediate results after each batch
        if results:
            temp_df = pd.DataFrame(all_features)
            temp_df.to_csv(f"{output_file}.part{i//batch_size + 1}", index=False)
            print(f"Saved {len(temp_df)} features to intermediate file")
    
    # Combine all the results
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_file, index=False)
        print(f"Extracted global thermodynamic features for {len(df)} RNA structures")
        print(f"Data saved to {output_file}")
        return df
    else:
        print("No features extracted.")
        return None

if __name__ == "__main__":
    dbn_directory = "dataset/bpRNA_1m_90_DBNFILES"
    output_file = "rna_global_thermo_features.csv"
    
    df = process_all_dbn_files(dbn_directory, output_file, max_files=None, num_workers=4)