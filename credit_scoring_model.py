import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def calculate_credit_score(input_file='user-wallet-transactions.json', output_file='wallet_scores.csv')
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # --- Feature Engineering ---
    
    wallet_activity = df.groupby('userWallet')['txHash'].count().to_frame('transaction_count')
    df['amount_usd'] = df['actionData'].apply(lambda x: float(x.get('amount', 0)) * float(x.get('assetPriceUSD', 0)))
    wallet_volume = df.groupby('userWallet')['amount_usd'].sum().to_frame('total_volume_usd')
    liquidation_counts = df[df['action'] == 'liquidationcall'].groupby('userWallet')['txHash'].count().to_frame('liquidation_count')
    repay_counts = df[df['action'] == 'repay'].groupby('userWallet')['txHash'].count().to_frame('repay_count')
    borrow_counts = df[df['action'] == 'borrow'].groupby('userWallet')['txHash'].count().to_frame('borrow_count')
    deposit_counts = df[df['action'] == 'deposit'].groupby('userWallet')['txHash'].count().to_frame('deposit_count')
    wallet_age = df.groupby('userWallet')['timestamp'].min().to_frame('first_seen')
    wallet_age['wallet_age_days'] = (pd.to_datetime('today') - wallet_age['first_seen']).dt.days
    wallet_age = wallet_age[['wallet_age_days']]

    # --- Combine Features ---
    wallets = pd.concat([
        wallet_activity,
        wallet_volume,
        liquidation_counts,
        repay_counts,
        borrow_counts,
        deposit_counts,
        wallet_age
    ], axis=1).fillna(0)

    # --- Scoring Logic (Heuristic-based) ---
    score_weights = {
        'transaction_count': 0.15,
        'total_volume_usd': 0.25,
        'liquidation_penalty': -0.40,
        'repay_reward': 0.30,
        'deposit_to_borrow_ratio': 0.20,
        'wallet_age_bonus': 0.10
    }

    scaler = MinMaxScaler()
    wallets_scaled = pd.DataFrame(scaler.fit_transform(wallets), columns=wallets.columns, index=wallets.index)

    # Calculate scores
    wallets['liquidation_penalty'] = wallets_scaled['liquidation_count'] * score_weights['liquidation_penalty']
    wallets['repay_reward'] = wallets_scaled['repay_count'] * score_weights['repay_reward']
    wallets['deposit_to_borrow_ratio'] = (wallets['deposit_count'] / (wallets['borrow_count'] + 1)) * score_weights['deposit_to_borrow_ratio']
    wallets['wallet_age_bonus'] = wallets_scaled['wallet_age_days'] * score_weights['wallet_age_bonus']
    
    # Combine scores and scale to 0-1000
    wallets['credit_score'] = (
        wallets_scaled['transaction_count'] * score_weights['transaction_count'] +
        wallets_scaled['total_volume_usd'] * score_weights['total_volume_usd'] +
        wallets['liquidation_penalty'] +
        wallets['repay_reward'] +
        wallets['deposit_to_borrow_ratio'] +
        wallets['wallet_age_bonus']
    )
    
    wallets['credit_score'] = (wallets['credit_score'] - wallets['credit_score'].min()) / (wallets['credit_score'].max() - wallets['credit_score'].min()) * 1000
    wallets['credit_score'] = wallets['credit_score'].astype(int)
    wallets[['credit_score']].to_csv(output_file)
    print(f"Credit scores calculated and saved to {output_file}")
    return wallets

if __name__ == '__main__':
    scored_wallets = calculate_credit_score()

    # --- Generate Analysis Graph ---
    plt.figure(figsize=(10, 6))
    plt.hist(scored_wallets['credit_score'], bins=range(0, 1001, 100), edgecolor='black')
    plt.title('Distribution of Wallet Credit Scores')
    plt.xlabel('Credit Score')
    plt.ylabel('Number of Wallets')
    plt.xticks(range(0, 1001, 100))
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('score_distribution.png')
    print("Score distribution graph saved to score_distribution.png")
