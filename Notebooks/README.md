# Example input

Place a small CSV here to test prediction. It must contain:

- a SMILES column named `Cano_Smile`, `SMILES`, or `PUBCHEM_EXT_DATASOURCE_SMILES`
- SwissADME descriptor columns (physicochemical + ADME), using the exact SwissADME
  export headers (e.g., MW, TPSA, iLOGP, XLOGP3, Consensus Log P, GI absorption,
  BBB permeant, Pgp substrate, CYP1A2/2C19/2C9/2D6/3A4 inhibitor, Synthetic Accessibility, ...)

SwissADME (http://www.swissadme.ch/) is a web tool; obtain the descriptor CSV there and
merge it with your SMILES before running `mtmm-predict` / `explain.py`.
