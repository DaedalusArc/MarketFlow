# trading_simulator.py - محاكاة تداول كاملة بدون API
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

class TradingSimulator:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = 0  # عدد الأسهم
        self.trades = []
        self.equity_curve = []
        
    def execute_trade(self, signal, price, date):
        """تنفيذ صفقة محاكاة"""
        # signal: -1 (بيع), 0 (لا شيء), 1 (شراء)
        
        if signal > 0.05 and self.positions == 0:  # شراء
            # استثمر 100% من رأس المال
            self.positions = self.capital / price
            self.capital = 0
            self.trades.append(('BUY', date, price, self.positions))
            return f"🟢 شراء {self.positions:.2f} سهم بسعر ${price:.2f}"
            
        elif signal < -0.05 and self.positions > 0:  # بيع
            self.capital = self.positions * price
            self.positions = 0
            self.trades.append(('SELL', date, price, self.capital))
            return f"🔴 بيع الكمية بسعر ${price:.2f} - الرصيد: ${self.capital:.2f}"
        
        return "⏸️ انتظار"
    
    def get_portfolio_value(self, current_price):
        """حساب قيمة المحفظة الحالية"""
        return self.capital + (self.positions * current_price)
    
    def get_return(self, current_price):
        """حساب العائد على الاستثمار"""
        current_value = self.get_portfolio_value(current_price)
        return ((current_value - self.initial_capital) / self.initial_capital) * 100

# تشغيل المحاكاة
print("🚀 بدء محاكاة التداول")
print("="*60)

# تحميل البيانات وتدريب النموذج (نفس الكود السابق)
from sklearn.preprocessing import StandardScaler

# [استخدم دوال compute_features, create_sequences, causal_tcn_model من الكود السابق]

# تدريب النموذج
print("📊 تدريب النموذج...")
data = yf.download('AAPL', start='2020-01-01', end='2024-12-31', progress=False)
features = compute_features(data)
prices = data['Close'].loc[features.index]

X, y = create_sequences(features, prices, window_size=64)

# تقسيم البيانات
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# تطبيع البيانات
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_test_flat = X_test.reshape(-1, X_test.shape[-1])
scaler.fit(X_train_flat)

X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

# بناء وتدريب النموذج
model = causal_tcn_model((64, 12))
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)

print("✅ النموذج مدرب!\n")

# محاكاة التداول
sim = TradingSimulator(initial_capital=100000)
print(f"💰 رأس المال الابتدائي: ${sim.initial_capital:,.2f}\n")

# اختبار النموذج على بيانات الاختبار
test_prices = prices.iloc[split_idx+64:].values
predictions = model.predict(X_test_scaled, verbose=0)

portfolio_values = []
signals = []

for i, pred in enumerate(predictions):
    signal = 2 * pred[0] - 1  # تحويل إلى -1..1
    signals.append(signal)
    
    # تنفيذ الصفقة
    result = sim.execute_trade(signal, test_prices[i], i)
    
    # حساب قيمة المحفظة
    portfolio_value = sim.get_portfolio_value(test_prices[i])
    portfolio_values.append(portfolio_value)
    
    # طباعة النتائج كل 20 صفقة
    if i % 50 == 0:
        print(f"📈 اليوم {i}: {result} | المحفظة: ${portfolio_value:,.2f}")

# النتائج النهائية
print("\n" + "="*60)
print("📊 النتائج النهائية:")
print("="*60)
final_value = portfolio_values[-1] if portfolio_values else sim.initial_capital
total_return = ((final_value - sim.initial_capital) / sim.initial_capital) * 100

print(f"💰 رأس المال الابتدائي: ${sim.initial_capital:,.2f}")
print(f"💰 القيمة النهائية: ${final_value:,.2f}")
print(f"📈 العائد الإجمالي: {total_return:+.2f}%")
print(f"📊 عدد الصفقات: {len(sim.trades)}")

# رسم النتائج
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label='قيمة المحفظة', color='green')
plt.axhline(y=sim.initial_capital, color='red', linestyle='--', label='رأس المال الابتدائي')
plt.title('أداء نظام التداول')
plt.xlabel('أيام التداول')
plt.ylabel('قيمة المحفظة ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()