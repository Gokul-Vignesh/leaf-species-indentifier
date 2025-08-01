
# main.py

import sys
from src.train import train
from src.evaluate import evaluate

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|eval]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "eval":
        evaluate()
    else:
        print("Unknown command:", cmd)
        sys.exit(1)

if __name__ == "__main__":
    main()

