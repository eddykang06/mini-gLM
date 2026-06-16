"""Functions to configure pre-training and fine-tuning data"""

import numpy as np
import pandas as pd
import subprocess
import sys
from pyfaidx import Fasta
from pathlib import Path


def run_cmd(command):
    result = subprocess.run(
        command,
        cwd = None,
        shell = False,
        text = True,
        capture_output = True
    )

    # Check for errors
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Command failed: {command}")

    return result.stdout


def configure_root():
    """
    Configure root directory structure based on Colab or local files
    """
    COLAB = Path("/content").exists()
    repo_url = "https://github.com/eddykang06/mini-gLM.git"
    repo_dir = Path("mini-gLM")

    if COLAB:
        
        root = Path("/content/mini-gLM")
        
        if not repo_dir.exists():
            run_cmd(["git", "clone", repo_url])

            # Create data folder
            data_dir = repo_dir / "data"
            data_dir.mkdir(parents = True, exist_ok = True)

            # Download files
            run_cmd([
                "curl", "-L", "-C", "-",
                "https://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
                "-o", str(data_dir / "hg38.fa.gz")
            ])

            run_cmd([
                "curl", "-L", "-C", "-",
                "https://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes",
                "-o", str(data_dir / "hg38.chrom.sizes")
            ])

            # Unzip hg38.fa.gz into hg38.fa
            hg38_gz = data_dir / "hg38.fa.gz"
            hg38_fa = data_dir / "hg38.fa"
            
            with open(hg38_fa, "w") as output_file:
                subprocess.run(
                    ["gunzip", "-c", str(hg38_gz)],
                    stdout = output_file,
                    check = True
                )
        
    else:
        root = Path.cwd().parent
    
    return root

def sample_from_fasta(
        data_dir: Path,
        out_dir: Path,
        n_seqs: int,
        min_length: int,
        max_length: int,
        seed: int | None = None,
) -> pd.DataFrame:
    """
    Function to sample random sequences of a specified length range from .fa file, weighted by chromsome length
    Final output is written to csv.

    Args:
        data_dir        : Path to dir containing data files
        out_dir         : Path to write final csv to
        n_seqs          : Number of sequences to sample
        min_length      : Minimum sequence length to sample
        max_length      : Maximum sequence length to sample
        seed            : Random seed for numpy
    
    Returns:
        csv : Dataframe of sampled sequences with start, end, length, and chromosome ID
    """
    # Store paths to data
    fasta_file = data_dir / "hg38.fa"
    chrom_size_file = data_dir / "hg38.chrom.sizes"

    # Set random seed if specified
    rng = np.random.default_rng(seed)

    # Store file using pyfaidx
    f = Fasta(fasta_file)

    # Load the chromosome and sort
    size_df = pd.read_table(chrom_size_file, sep = "\t", index_col = 0, header = None)
    size_df.columns = ["length"]
    size_df["length"] = size_df["length"].astype(int)
    size_df = size_df.iloc[size_df.index.argsort()]

    # Check that the fasta chromsomes match the size df
    index = sorted(size_df.index.to_numpy())
    chromosomes = sorted(np.array(list(f.keys())))

    if index != chromosomes:
        raise KeyError("Fasta chromosomes IDs and length data IDs are incompatible")

    # Normalize to obtain sampling weights for each chromosome
    weights = size_df["length"].to_numpy() / size_df["length"].to_numpy().sum()

    # Initiate list to store sequences and metadata
    seqs = []
    chroms = []
    starts = []
    ends = []
    lengths = []

    # Counter so that rejected sequences don't count toward total count
    counter = 0

    # Number of sequences to sample
    while counter < n_seqs:

        # Weighted sampling of chromosome
        chrom = rng.choice(chromosomes, p = weights)
        length = int(size_df.loc[chrom, "length"])

        # Randomly sample a sequence
        seq_length = rng.integers(min_length, min(length, max_length) + 1)
        start_idx = rng.integers(0, length - seq_length + 1)
        seq = f[chrom][start_idx:start_idx + seq_length]
        seq = str(seq).upper()

        # Split across "N" character if exists and take first substr
        seq = seq.split("N")[0]

        # Convert to uppercase, remove non-alpha and non-ACGT characters 
        seq = "".join(char for char in seq if char in "ACGT")
        new_length = len(seq)

        # Check that sequences is still above minimum length
        if new_length >= min_length:
            counter += 1 

            # Append sequence and all metadata
            seqs.append(seq)
            chroms.append(chrom)
            starts.append(start_idx)
            ends.append(start_idx + seq_length)
            lengths.append(new_length)

    # Construct dataframe
    df = pd.DataFrame({
        "sequence": seqs,
        "chromosome_id": chroms,
        "start": starts,
        "end": ends,
        "length": lengths
    })

    # Write to csv
    out_file = out_dir / "pretraining.csv.gz"
    df.to_csv(out_file, compression = "gzip", index = False)