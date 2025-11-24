# RFLPA Implementation

A Python implementation of **RFLPA: A Robust Federated Learning Framework against Poisoning Attacks with Secure Aggregation** based on the NeurIPS 2024 paper.

## Overview

This implementation provides a privacy-preserving federated learning framework that combines secure aggregation with robust defense mechanisms against poisoning attacks. The framework uses cryptographic techniques to protect client privacy while maintaining model robustness against malicious participants.

## Key Features

- **Secure Aggregation**: Uses packed Shamir secret sharing to protect client gradients
- **Poisoning Attack Defense**: Implements cosine similarity-based trust scoring
- **Privacy Preservation**: Ensures server cannot access individual client updates
- **Efficient Communication**: Reduces overhead with packed secret sharing
- **Cryptographic Security**: Includes encryption, signatures, and key exchange

## Project Structure
.....





## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Jeddou10/RFLPA-Reimplementation.git
    cd rflpa_implementation
2. Install dependencies:
    ```bash
    pip install -r requirements.txt

