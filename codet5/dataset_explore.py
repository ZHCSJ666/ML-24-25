


#!/usr/bin/env python
# -*- coding: utf-8 -*-


from datasets import load_dataset

def main():
    dataset = load_dataset("Rissou/simplified-commit-chronicle")
    print(dataset)


    print("\n=== next sample ===")
    print(dataset["train"][0])

if __name__ == "__main__":
    main()
