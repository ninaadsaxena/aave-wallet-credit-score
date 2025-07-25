# Aave Wallet Credit Scoring Model

This project provides a machine learning model to assign a credit score to wallets on the Aave V2 protocol. The score, ranging from 0 to 1000, is based on a wallet's historical transaction behavior. Higher scores indicate more reliable and responsible usage of the protocol, while lower scores may reflect riskier or bot-like behavior.

---

## Methodology

Due to the absence of labeled data (i.e., pre-existing, verified credit scores), this model employs an unsupervised, heuristic-based approach. We engineer several features from the raw transaction data that are indicative of a user's financial health and behavior within the Aave protocol. Each feature is assigned a weight based on its perceived importance in determining creditworthiness. These weighted features are then combined to produce a final credit score.

---

## Architecture

The model is implemented in a single Python script (`credit_scoring_model.py`) that performs the following steps:

### 1. Data Loading and Preprocessing

- Reads the raw transaction data from a JSON file (`user-wallet-transactions.json`).
- Converts Unix timestamps to readable datetime objects.
- Parses `amount` and `assetPriceUSD` strings into numeric types.
- Normalizes amounts based on token decimal precision (e.g., USDC has 6 decimals; WETH/DAI have 18).
- Calculates `amount_usd` for each transaction using `amount × assetPriceUSD`.

### 2. Feature Engineering

Aggregates transaction data for each unique `userWallet`, calculating:

- **Financial Activity**: Total USD value of deposits, borrows, repays, and redemptions.
- **Risk Indicators**: Number of `liquidationcall` events.
- **Engagement Metrics**: Total number of transactions, duration of activity (first to last transaction), number of unique assets.
- **Behavioral Ratios**:
  - `repay_to_borrow_ratio`: (total repays / total borrows), capped at 1.0
  - `activity_score`: (transactions per day of activity)

### 3. Credit Score Calculation

- Each feature is min-max normalized (scaled to 0–1).
- Predefined `FEATURE_WEIGHTS` are applied:
  - Positive weights → increase score
  - Negative weights → decrease score
- Weighted sum is scaled to a final **0–1000** score range.
- Scores are clipped to ensure they're within bounds.

### 4. Output Generation

- Final scores saved to `wallet_scores.csv`.

### 5. Analysis and Visualization

- Basic descriptive statistics on the generated scores.
- Histogram of score distribution across 100-point bins (e.g., 0–99, 100–199, etc.).
- Example features printed for:
  - Top 5% wallets
  - Bottom 5% wallets

---

## Processing Flow

### Load Data

The script starts by loading the `user-wallet-transactions.json` file into a pandas DataFrame.

### Feature Engineering

The following features are engineered for each wallet:

- **Transaction Frequency**: Total number of transactions.  
- **Total Transaction Volume (USD)**: The sum of the value of all transactions in USD.  
- **Liquidation History**: The number of times a wallet has been liquidated.  
- **Repayment History**: The number of times a wallet has repaid a loan.  
- **Deposit-to-Borrow Ratio**: The ratio of deposits to borrows, indicating how much of their borrowing is backed by their own assets.  
- **Wallet Age**: The number of days since the wallet's first transaction.

---

## Scoring Logic

- The features are normalized to a 0–1 scale to ensure fair comparison.
- A weighted sum of the normalized features is calculated to generate a raw score. The weights are defined as follows:

  | Feature                  | Weight  |
  |--------------------------|---------|
  | Transaction Count        | 15%     |
  | Total Volume (USD)       | 25%     |
  | Liquidation Penalty      | -40%    |
  | Repayment Reward         | 30%     |
  | Deposit-to-Borrow Ratio  | 20%     |
  | Wallet Age Bonus         | 10%     |

- The final credit score is then scaled to a range of **0 to 1000**.

## Author

Developed by **Ninaad Saxena**. Feel free to connect on [LinkedIn](https://www.linkedin.com/in/ninaadsaxena/).

---

