#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Day-by-Day Analysis and Visualization
--------------------------------------
Generates daily statistics and visual charts from scalp_with_trend_results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Configuration
RESULTS_CSV = "scalp_with_trend_results.csv"
OUTPUT_IMAGE = "daily_analysis.png"

def analyze_daily():
    # Load trades
    trades = pd.read_csv(RESULTS_CSV, parse_dates=['entry_time', 'exit_time'])

    # Extract date from exit_time for daily grouping
    trades['date'] = trades['exit_time'].dt.date

    # Group by date
    daily_stats = []

    for date, group in trades.groupby('date'):
        total_trades = len(group)
        wins = (group['pnl_rupees'] > 0).sum()
        losses = (group['pnl_rupees'] < 0).sum()
        breakeven = (group['pnl_rupees'] == 0).sum()

        total_pnl = group['pnl_rupees'].sum()
        win_pnl = group.loc[group['pnl_rupees'] > 0, 'pnl_rupees'].sum()
        loss_pnl = group.loc[group['pnl_rupees'] < 0, 'pnl_rupees'].sum()

        daily_stats.append({
            'date': date,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'breakeven': breakeven,
            'total_pnl': total_pnl,
            'win_pnl': win_pnl,
            'loss_pnl': loss_pnl
        })

    df_daily = pd.DataFrame(daily_stats)
    df_daily['cumulative_pnl'] = df_daily['total_pnl'].cumsum()

    # Print summary
    print("=" * 100)
    print("📅 DAY-BY-DAY ANALYSIS")
    print("=" * 100)
    print(f"\n{'Date':<12} {'Trades':>8} {'Wins':>6} {'Loss':>6} {'B/E':>6} {'Daily P&L':>12} {'Cum P&L':>12}")
    print("-" * 100)

    for _, row in df_daily.iterrows():
        print(f"{str(row['date']):<12} {row['total_trades']:>8} {row['wins']:>6} {row['losses']:>6} "
              f"{row['breakeven']:>6} ₹{row['total_pnl']:>10,.0f} ₹{row['cumulative_pnl']:>10,.0f}")

    print("-" * 100)
    print(f"{'TOTAL':<12} {df_daily['total_trades'].sum():>8} {df_daily['wins'].sum():>6} "
          f"{df_daily['losses'].sum():>6} {df_daily['breakeven'].sum():>6} "
          f"₹{df_daily['total_pnl'].sum():>10,.0f} ₹{df_daily['cumulative_pnl'].iloc[-1]:>10,.0f}")
    print("=" * 100)

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Day-by-Day Trading Performance', fontsize=16, fontweight='bold')

    dates = df_daily['date']

    # Plot 1: Daily P&L
    ax1 = axes[0]
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in df_daily['total_pnl']]
    ax1.bar(dates, df_daily['total_pnl'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_ylabel('Daily P&L (₹)', fontsize=12, fontweight='bold')
    ax1.set_title('Daily Profit/Loss', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Cumulative P&L
    ax2 = axes[1]
    ax2.plot(dates, df_daily['cumulative_pnl'], marker='o', linewidth=2,
             markersize=6, color='blue', label='Cumulative P&L')
    ax2.fill_between(dates, df_daily['cumulative_pnl'], alpha=0.3, color='blue')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('Cumulative P&L (₹)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Profit/Loss Over Time', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Trades breakdown (Wins vs Losses)
    ax3 = axes[2]
    x = range(len(dates))
    width = 0.35

    ax3.bar([i - width/2 for i in x], df_daily['wins'], width,
            label='Wins', color='green', alpha=0.7, edgecolor='black')
    ax3.bar([i + width/2 for i in x], df_daily['losses'], width,
            label='Losses', color='red', alpha=0.7, edgecolor='black')

    ax3.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.set_title('Daily Wins vs Losses', fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(d) for d in dates], rotation=45, ha='right')
    ax3.legend(loc='upper left')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {OUTPUT_IMAGE}")
    print(f"📊 Open the image to view the charts")

    return df_daily

if __name__ == "__main__":
    analyze_daily()
