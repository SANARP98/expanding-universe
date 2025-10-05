#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest Web Application Server
--------------------------------
Flask-based single-page web application for running the complete backtest workflow:
1. Fetch historical data
2. Run backtest
3. Analyze results

Access at http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import subprocess
import os
import base64
import json
from datetime import datetime

app = Flask(__name__)

# HTML Template (embedded)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }

        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .content {
            padding: 30px;
        }

        .section {
            margin-bottom: 25px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }

        .section-header {
            background: #f5f5f5;
            padding: 15px 20px;
            cursor: pointer;
            font-weight: bold;
            font-size: 1.1em;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.3s;
        }

        .section-header:hover {
            background: #eeeeee;
        }

        .section-header.active {
            background: #667eea;
            color: white;
        }

        .section-content {
            padding: 20px;
            display: none;
        }

        .section-content.active {
            display: block;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: 600;
            margin-bottom: 5px;
            color: #333;
            font-size: 0.9em;
        }

        input[type="text"],
        input[type="number"],
        input[type="date"],
        select {
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 1em;
            transition: border-color 0.3s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }

        input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }

        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .run-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.2em;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 20px;
            width: 100%;
        }

        .run-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .run-button:active {
            transform: translateY(0);
        }

        .run-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        #status {
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            display: none;
            font-weight: 500;
        }

        #status.info {
            background: #e3f2fd;
            color: #1976d2;
            border: 1px solid #90caf9;
        }

        #status.success {
            background: #e8f5e9;
            color: #388e3c;
            border: 1px solid #81c784;
        }

        #status.error {
            background: #ffebee;
            color: #d32f2f;
            border: 1px solid #ef5350;
        }

        #results {
            margin-top: 30px;
        }

        .results-section {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }

        .results-section h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }

        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.4;
        }

        img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .time-window-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .time-window-group input {
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 Scalp-with-Trend Backtest Dashboard</h1>
            <p>Configure, Run, and Analyze Your Trading Strategy</p>
        </header>

        <div class="content">
            <form id="backtestForm">
                <!-- Data Fetch Section -->
                <div class="section">
                    <div class="section-header active" onclick="toggleSection(this)">
                        📊 Data Fetch Configuration
                        <span>▼</span>
                    </div>
                    <div class="section-content active">
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Symbol</label>
                                <input type="text" name="symbol" value="NIFTY25NOV2525800PE" required>
                            </div>
                            <div class="form-group">
                                <label>Exchange</label>
                                <select name="exchange">
                                    <option value="NFO">NFO</option>
                                    <option value="NSE">NSE</option>
                                    <option value="BSE">BSE</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Interval</label>
                                <select name="interval">
                                    <option value="1m">1 Minute</option>
                                    <option value="5m" selected>5 Minutes</option>
                                    <option value="15m">15 Minutes</option>
                                    <option value="1h">1 Hour</option>
                                    <option value="D">Daily</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Start Date</label>
                                <input type="date" name="start_date" value="2025-07-01" required>
                            </div>
                            <div class="form-group">
                                <label>End Date</label>
                                <input type="date" name="end_date" value="2025-10-04" required>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Capital & Position Sizing -->
                <div class="section">
                    <div class="section-header" onclick="toggleSection(this)">
                        💰 Capital & Position Sizing
                        <span>▼</span>
                    </div>
                    <div class="section-content">
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Starting Capital (₹)</label>
                                <input type="number" name="starting_capital" value="200000" required>
                            </div>
                            <div class="form-group">
                                <label>Quantity Per Point</label>
                                <input type="number" name="qty_per_point" value="150" required>
                            </div>
                            <div class="form-group">
                                <label>Brokerage Per Trade Leg (₹)</label>
                                <input type="number" name="brokerage_per_trade" value="20" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>Slippage Per Leg (Points)</label>
                                <input type="number" name="slippage_points" value="0.10" step="0.01" required>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Targets & Stops -->
                <div class="section">
                    <div class="section-header" onclick="toggleSection(this)">
                        🎯 Targets & Stop Loss
                        <span>▼</span>
                    </div>
                    <div class="section-content">
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Target Points</label>
                                <input type="number" name="target_points" value="10" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>Stop Loss Points</label>
                                <input type="number" name="stoploss_points" value="2" step="0.1" required>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Indicators -->
                <div class="section">
                    <div class="section-header" onclick="toggleSection(this)">
                        📈 Indicators & Filters
                        <span>▼</span>
                    </div>
                    <div class="section-content">
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Fast EMA Period</label>
                                <input type="number" name="ema_fast" value="5" required>
                            </div>
                            <div class="form-group">
                                <label>Slow EMA Period</label>
                                <input type="number" name="ema_slow" value="20" required>
                            </div>
                            <div class="form-group">
                                <label>ATR Window</label>
                                <input type="number" name="atr_window" value="14" required>
                            </div>
                            <div class="form-group">
                                <label>ATR Min Points</label>
                                <input type="number" name="atr_min_points" value="2.0" step="0.1" required>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Advanced Exit Logic -->
                <div class="section">
                    <div class="section-header" onclick="toggleSection(this)">
                        🔬 Advanced Exit Logic
                        <span>▼</span>
                    </div>
                    <div class="section-content">
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Exit Bar Path Model</label>
                                <select name="exit_bar_path">
                                    <option value="color" selected>Color-based (Green/Red bars)</option>
                                    <option value="bull">Bullish (Always low→high)</option>
                                    <option value="bear">Bearish (Always high→low)</option>
                                    <option value="worst">Worst Case (Conservative)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <div class="checkbox-group">
                                    <input type="checkbox" name="confirm_trend_at_entry" checked>
                                    <label>Confirm Trend at Entry (Conservative)</label>
                                </div>
                            </div>
                            <div class="form-group">
                                <div class="checkbox-group">
                                    <input type="checkbox" name="enable_eod_square_off" checked>
                                    <label>Enable EOD Square-off (Intraday Only)</label>
                                </div>
                            </div>
                            <div class="form-group">
                                <label>Square-off Time (IST)</label>
                                <input type="time" name="square_off_time" value="15:25" required>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Session & Risk Management -->
                <div class="section">
                    <div class="section-header" onclick="toggleSection(this)">
                        ⏰ Session Windows & Risk Management
                        <span>▼</span>
                    </div>
                    <div class="section-content">
                        <div class="form-group" style="margin-bottom: 15px;">
                            <label>Session Window 1</label>
                            <div class="time-window-group">
                                <input type="time" name="session1_start" value="10:20" required>
                                <span>to</span>
                                <input type="time" name="session1_end" value="11:00" required>
                            </div>
                        </div>
                        <div class="form-group" style="margin-bottom: 15px;">
                            <label>Session Window 2</label>
                            <div class="time-window-group">
                                <input type="time" name="session2_start" value="11:01" required>
                                <input type="time" name="session2_end" value="14:29" required>
                            </div>
                        </div>
                        <div class="form-group">
                            <label>Daily Loss Cap (₹)</label>
                            <input type="number" name="daily_loss_cap" value="-1000" required>
                        </div>
                    </div>
                </div>

                <button type="submit" class="run-button" id="runButton">
                    ▶️ Run Complete Backtest
                </button>
            </form>

            <div class="spinner" id="spinner"></div>

            <div id="status"></div>

            <div id="results"></div>
        </div>
    </div>

    <script>
        function toggleSection(header) {
            const content = header.nextElementSibling;
            const arrow = header.querySelector('span');

            header.classList.toggle('active');
            content.classList.toggle('active');

            if (content.classList.contains('active')) {
                arrow.textContent = '▼';
            } else {
                arrow.textContent = '▶';
            }
        }

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = type;
            status.style.display = 'block';
        }

        function hideStatus() {
            document.getElementById('status').style.display = 'none';
        }

        document.getElementById('backtestForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(e.target);
            const data = {};

            for (let [key, value] of formData.entries()) {
                if (key.includes('enable_') || key.includes('disable_') || key.includes('confirm_')) {
                    data[key] = true;
                } else {
                    data[key] = value;
                }
            }

            // Handle unchecked checkboxes
            if (!formData.has('confirm_trend_at_entry')) data.confirm_trend_at_entry = false;
            if (!formData.has('enable_eod_square_off')) data.enable_eod_square_off = false;

            // Disable button and show spinner
            const runButton = document.getElementById('runButton');
            const spinner = document.getElementById('spinner');
            runButton.disabled = true;
            runButton.textContent = '⏳ Running...';
            spinner.style.display = 'block';

            showStatus('🔄 Starting backtest pipeline...', 'info');
            document.getElementById('results').innerHTML = '';

            try {
                const response = await fetch('/api/run_backtest', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.success) {
                    showStatus('✅ Backtest completed successfully!', 'success');

                    let html = '';

                    // Backtest output
                    if (result.backtest_output) {
                        html += `
                            <div class="results-section">
                                <h3>📊 Backtest Summary</h3>
                                <pre>${result.backtest_output}</pre>
                            </div>
                        `;
                    }

                    // Analysis output
                    if (result.analysis_output) {
                        html += `
                            <div class="results-section">
                                <h3>📅 Day-by-Day Analysis</h3>
                                <pre>${result.analysis_output}</pre>
                            </div>
                        `;
                    }

                    // Chart image
                    if (result.chart_image) {
                        html += `
                            <div class="results-section">
                                <h3>📈 Performance Charts</h3>
                                <img src="data:image/png;base64,${result.chart_image}" alt="Performance Charts">
                            </div>
                        `;
                    }

                    document.getElementById('results').innerHTML = html;
                } else {
                    showStatus('❌ Error: ' + result.error, 'error');
                }
            } catch (error) {
                showStatus('❌ Error: ' + error.message, 'error');
            } finally {
                runButton.disabled = false;
                runButton.textContent = '▶️ Run Complete Backtest';
                spinner.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/run_backtest', methods=['POST'])
def run_backtest():
    """Run the complete backtest pipeline"""
    try:
        data = request.json

        # Step 1: Fetch historical data
        symbol = data.get('symbol', 'NIFTY25NOV2525800PE')
        exchange = data.get('exchange', 'NFO')
        interval = data.get('interval', '5m')
        start_date = data.get('start_date', '2025-07-01')
        end_date = data.get('end_date', '2025-10-04')

        # Run fetch_history.py
        fetch_cmd = [
            'python3', 'fetch_history.py',
            '--symbol', symbol,
            '--exchange', exchange,
            '--interval', interval,
            '--start_date', start_date,
            '--end_date', end_date
        ]

        fetch_result = subprocess.run(fetch_cmd, capture_output=True, text=True)
        if fetch_result.returncode != 0:
            return jsonify({
                'success': False,
                'error': f'Data fetch failed: {fetch_result.stderr}'
            })

        # Step 2: Run backtest
        input_csv = f"{symbol}_history.csv"

        backtest_cmd = [
            'python3', 'aaa2.py',
            '--input_csv', input_csv,
            '--starting_capital', str(data.get('starting_capital', 200000)),
            '--qty_per_point', str(data.get('qty_per_point', 150)),
            '--target_points', str(data.get('target_points', 10)),
            '--stoploss_points', str(data.get('stoploss_points', 2)),
            '--ema_fast', str(data.get('ema_fast', 5)),
            '--ema_slow', str(data.get('ema_slow', 20)),
            '--atr_window', str(data.get('atr_window', 14)),
            '--atr_min_points', str(data.get('atr_min_points', 2.0)),
            '--daily_loss_cap', str(data.get('daily_loss_cap', -1000)),
            '--brokerage_per_trade', str(data.get('brokerage_per_trade', 20.0)),
            '--slippage_points', str(data.get('slippage_points', 0.10)),
            '--exit_bar_path', str(data.get('exit_bar_path', 'color')),
            '--confirm_trend_at_entry', str(data.get('confirm_trend_at_entry', True)),
            '--enable_eod_square_off', str(data.get('enable_eod_square_off', True)),
            '--square_off_time', str(data.get('square_off_time', '15:25'))
        ]

        backtest_result = subprocess.run(backtest_cmd, capture_output=True, text=True)
        if backtest_result.returncode != 0:
            return jsonify({
                'success': False,
                'error': f'Backtest failed: {backtest_result.stderr}'
            })

        # Step 3: Run analysis
        analysis_cmd = ['python3', 'analyze_daybyday.py']
        analysis_result = subprocess.run(analysis_cmd, capture_output=True, text=True)

        # Read and encode the chart image
        chart_image = None
        if os.path.exists('daily_analysis.png'):
            with open('daily_analysis.png', 'rb') as img_file:
                chart_image = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({
            'success': True,
            'backtest_output': backtest_result.stdout,
            'analysis_output': analysis_result.stdout if analysis_result.returncode == 0 else None,
            'chart_image': chart_image
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("🚀 Starting Backtest Dashboard Server...")
    print("📡 Access the application at: http://localhost:1111")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=1111)
