# AI Agent Guidance for TDEM_Thesis_Yurek

## Purpose
This workspace contains Python scripts and Jupyter notebooks for transient electromagnetic (TEM) survey processing, waveform analysis, forward modeling, and data parsing for a hydrogeophysics thesis project.

## Key files
- `WAV_processor_V3.py`: TEM pulse processing pipeline. Uses audio files, pulse detection, segment alignment, log-gated decay computation, and plotting.
- `usfParse.py`: Parser for `.usf` TEM station files. Produces pandas DataFrames, filters quality, summarizes data, and exports results to Excel/CSV.
- `SimPEG_DTDEM_ForwardModel.py`: 1D layered conductivity forward model using SimPEG. Generates synthetic decay responses and includes plotting helpers.
- `Forward_Modeling_TDEM.ipynb`: Notebook documenting forward modeling experiments and inversion-related notes.
- `HYDRUS_Procesing.ipynb`: Notebook for HYDRUS-related processing.

## What agents should know
- The repository is small and script-driven; focus on clarity, portability, and modular refactoring rather than full application architecture.
- Many scripts contain hard-coded Windows paths and direct top-level execution blocks. Prefer improving usability with command-line arguments, relative paths, and reusable functions.
- There is no centralized packaging or dependency manifest. Use standard scientific Python libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `simpeg`.
- The notebooks are exploratory and may include analysis notes rather than production-ready code.

## Recommended agent behavior
- Preserve existing script intentions when restructuring code. Do not remove or replace scientific logic without confirming expected outputs.
- Avoid assuming external data files exist unless the user provides them. When adding examples, use placeholders or describe required input formats.
- Use the repository’s domain terms: TEM, USF, WAV pulse, conductivity, decay gates, forward modeling, SimPEG, layered models.
- When asked to enhance the code, prioritize:
  1. replacing hard-coded user-specific file paths with arguments/configuration,
  2. making command-line behavior explicit,
  3. keeping analysis notebooks readable and reproducible.

## Notes for future customization
- If project scope grows, consider adding `.github/copilot-instructions.md` with runtime info and a `requirements.txt` for dependency management.
- If there are multiple domains later (e.g. waveform processing, inversion, HYDRUS), split guidance into domain-specific agent instructions.
