### EDMD implementation

### Connect to github

1. pull changes
   ```bash
    git pull origin main
   ```
   
2. push changes
   ```bash
    git add .
    git commit -m "Descrizione delle modifiche"
    git push origin main
   ```
   
### Prerequisites

1. Dependencies listed in `requirements.txt`

### Installation

1. install dependencies:
   ```bash
    pip install -r requirements.txt
   ```
### How to run the code
1. trajectory optimization:
   ```bash
    python3 src/TO.py
   ```
2.  motion tracking vers-1:
   ```bash
    python3 src/MPC.py
   ```
