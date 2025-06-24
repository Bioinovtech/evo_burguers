# Evo Burgers

## Setup

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/) installed on your system.

### Environment Setup

1. Clone the repository:

   If you haven't cloned the repository yet, run:
   
   ```bash
   git clone --recursive <repository-url>
   ```

   If you've already cloned the repository without submodules, run:
   
   ```bash
   git submodule update --init --recursive
   ```

3. Create the Conda environment using the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

4. Activate the environment:
    ```bash
    conda activate ldm
    ```

## Usage

The main script for running the project is `token_ev_ga.py`. Below are the arguments it accepts:

### Arguments

- `--seed` (int): Seed for random number generation (for a single run).
- `--seed_path` (str): Path to the seed list txt file (for multiple runs).
- `--cuda` (int): CUDA GPU to use.
- `--predictor` (int): Aesthetic predictor to use:
  - `0`: SAM
  - `1`: LAION

### Example Command

```bash
python token_ev_ga.py --seed 42 --cuda 0 --predictor 1
```
