import ccxt
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import sqlite3
import logging
import colorlog
import telegram
import os
import asyncio
import pickle
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

# Cấu hình colorlog
logger = colorlog.getLogger('bot')
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Tắt GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load biến môi trường
load_dotenv()
bot = telegram.Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Kết nối Binance
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET'),
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})

# Tham số toàn cục
SYMBOL = 'ETH/USDT'
LEVERAGE = 20
BASE_AMOUNT = 40
RISK_PER_TRADE_PERCENT = 0.01
STOP_LOSS_PERCENT = 0.015
TAKE_PROFIT_PERCENT = 0.01
TRADING_FEE_PERCENT = 0.0002
VOLATILITY_RISK_THRESHOLD = 0.02
LSTM_WINDOW = 60
DB_PATH = 'trade_predictions.db'
BREAKOUT_THRESHOLD = 1.001
BREAKDOWN_THRESHOLD = 0.9995
VOLATILITY_THRESHOLD = 0.01
DAILY_LOSS_LIMIT = 20
MAX_DAILY_TRADES = 30
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_SPIKE_THRESHOLD = 1.5
MACD_FAST = 6
MACD_SLOW = 13
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ADX_PERIOD = 14
MIN_PREDICTED_CHANGE = 0.0005
ADX_SIDWAYS_THRESHOLD = 25
RETRAIN_INTERVAL = 300  # Giảm xuống 5 phút
BUFFER_SIZE = 1000
CHECK_INTERVAL = 30
TRAILING_STOP_PERCENT = 0.01
STOCHASTIC_PERIOD = 14
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKO = 52
ENABLE_PNL_CHECK = True
PNL_THRESHOLD = 4.0
PNL_CHECK_INTERVAL = 900
MIN_CONFIDENCE = 0.7  # Ngưỡng xác suất thắng tối thiểu để vào lệnh
MAX_PREDICTION_ERROR = 0.05  # Sai số dự đoán tối đa 5%
BUY_THRESHOLD = 70
SELL_THRESHOLD = 70

# Biến toàn cục
scaler = MinMaxScaler()
lstm_model = None
rf_classifier = None
performance = {'profit': 0, 'trades': 0, 'total_profit': 0, 'total_loss': 0, 'consecutive_losses': 0, 'win_rate': 0}
is_trading = False
daily_trades = 0
position = None
data_buffer = []
last_retrain_time = time.time()
last_check_time = time.time()
last_pnl_check_time = time.time()

# --- Hàm khởi tạo mô hình ---
def create_lstm_model():
    inputs = Input(shape=(LSTM_WINDOW, 9))
    x = LSTM(100, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(100)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def create_rf_classifier():
    return RandomForestClassifier(n_estimators=100, random_state=42)

# --- Hàm cơ sở dữ liệu ---
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                predicted_price REAL,
                actual_price REAL,
                entry_price REAL,
                predicted_change REAL,
                actual_change REAL,
                profit REAL,
                strategy TEXT,
                buy_score REAL DEFAULT 0,
                sell_score REAL DEFAULT 0
            )''')
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(predictions)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'buy_score' not in columns:
                conn.execute("ALTER TABLE predictions ADD COLUMN buy_score REAL DEFAULT 0")
            if 'sell_score' not in columns:
                conn.execute("ALTER TABLE predictions ADD COLUMN sell_score REAL DEFAULT 0")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo cơ sở dữ liệu: {e}")

# --- Hàm huấn luyện mô hình ---
async def train_advanced_model(ohlcv, historical_closes, historical_highs, historical_lows, initial=False):
    global data_buffer, lstm_model, performance, scaler, rf_classifier
    try:
        if len(ohlcv) < LSTM_WINDOW + 10:
            logger.warning("Dữ liệu không đủ để huấn luyện mô hình")
            return

        closes = np.array([x[4] for x in ohlcv])
        volumes = np.array([x[5] for x in ohlcv])
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])

        atr = np.array([max(h - l, abs(h - closes[i-1]), abs(l - closes[i-1])) if i > 0 else h - l 
                        for i, (h, l) in enumerate(zip(highs, lows))])
        rsi = np.array([calculate_rsi(closes[:i+1]) or 50 for i in range(len(closes))])
        macd, _, _ = calculate_macd(closes) or (np.zeros(len(closes)), 0, 0)
        stochastic_result = calculate_stochastic(highs, lows, closes)
        stochastic = stochastic_result if stochastic_result is not None else np.full(len(closes), 50)
        ichimoku_result = calculate_ichimoku(highs, lows, closes)
        ichimoku = ichimoku_result if ichimoku_result is not None else np.zeros(len(closes))

        min_length = min(len(closes), len(atr), len(rsi), len(macd), len(stochastic), len(ichimoku))
        closes = closes[-min_length:]
        volumes = volumes[-min_length:]
        atr = atr[-min_length:]
        rsi = rsi[-min_length:]
        macd = macd[-min_length:]
        stochastic = stochastic[-min_length:]
        ichimoku = ichimoku[-min_length:]

        X, y = [], []
        for i in range(LSTM_WINDOW, len(closes)):
            ema_short = np.mean(closes[max(0, i-5):i+1])
            ema_long = np.mean(closes[max(0, i-15):i+1])
            data = np.column_stack((
                closes[i-LSTM_WINDOW:i],
                volumes[i-LSTM_WINDOW:i],
                np.full(LSTM_WINDOW, ema_short),
                np.full(LSTM_WINDOW, ema_long),
                atr[i-LSTM_WINDOW:i],
                rsi[i-LSTM_WINDOW:i],
                macd[i-LSTM_WINDOW:i],
                stochastic[i-LSTM_WINDOW:i],
                ichimoku[i-LSTM_WINDOW:i]
            ))
            if i == LSTM_WINDOW:
                scaler.fit(data)
                with open('scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
            X.append(scaler.transform(data))
            y.append(closes[i])
        X = np.array(X)
        y = np.array(y)

        if len(X) < 2:
            logger.warning("Không đủ mẫu dữ liệu để huấn luyện mô hình LSTM")
            return

        lstm_model.fit(X, y, epochs=20 if not initial else 50, batch_size=32, validation_split=0.1, verbose=0)
        lstm_model.save('lstm_model.keras')

        with sqlite3.connect(DB_PATH) as conn:
            trades = conn.execute("SELECT buy_score, sell_score, profit FROM predictions WHERE profit IS NOT NULL").fetchall()
            if len(trades) > 10:
                X_rf = np.array([[t[0], t[1]] for t in trades])
                y_rf = np.array([1 if t[2] > 0 else 0 for t in trades])
                rf_classifier.fit(X_rf, y_rf)
                with open('rf_classifier.pkl', 'wb') as f:
                    pickle.dump(rf_classifier, f)

        trades = conn.execute("SELECT profit FROM predictions WHERE profit IS NOT NULL").fetchall()
        wins = sum(1 for t in trades if t[0] > 0)
        performance['win_rate'] = wins / len(trades) if trades else 0
        logger.info(f"Đã {'huấn luyện ban đầu' if initial else 'cập nhật'} mô hình. Win Rate: {performance['win_rate']:.2f}")
    except Exception as e:
        logger.error(f"Lỗi huấn luyện mô hình: {e}")

# --- Các hàm chỉ báo kỹ thuật ---
def calculate_stochastic(highs, lows, closes, period=STOCHASTIC_PERIOD):
    if len(closes) < period + 1:
        return None
    k_values = []
    for i in range(period - 1, len(closes)):
        highest_high = np.max(highs[i-period+1:i+1])
        lowest_low = np.min(lows[i-period+1:i+1])
        k = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low) if (highest_high - lowest_low) != 0 else 50
        k_values.append(k)
    return np.array(k_values)

def calculate_ichimoku(highs, lows, closes):
    if len(closes) < ICHIMOKU_SENKO:
        return None
    tenkan_sen = np.array([(np.max(highs[i-ICHIMOKU_TENKAN+1:i+1]) + np.min(lows[i-ICHIMOKU_TENKAN+1:i+1])) / 2 
                          for i in range(ICHIMOKU_TENKAN-1, len(closes))])
    kijun_sen = np.array([(np.max(highs[i-ICHIMOKU_KIJUN+1:i+1]) + np.min(lows[i-ICHIMOKU_KIJUN+1:i+1])) / 2 
                          for i in range(ICHIMOKU_KIJUN-1, len(closes))])
    senkou_b = np.array([(np.max(highs[i-ICHIMOKU_SENKO+1:i+1]) + np.min(lows[i-ICHIMOKU_SENKO+1:i+1])) / 2 
                         for i in range(ICHIMOKU_SENKO-1, len(closes))])
    ichimoku = np.column_stack((tenkan_sen[-len(senkou_b):], kijun_sen[-len(senkou_b):], senkou_b)).mean(axis=1)
    return ichimoku if ichimoku.size > 0 else None

def calculate_rsi(closes, period=RSI_PERIOD):
    if len(closes) < period + 1:
        return None
    gains = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(closes, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    if len(closes) < slow + signal:
        return None, None, None
    def ema(data, period):
        alpha = 2 / (period + 1)
        ema_values = [data[0]]
        for i in range(1, len(data)):
            ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[-1])
        return np.array(ema_values)
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd = ema_fast - ema_slow
    signal_line = ema(macd[-len(macd)+signal-1:], signal)[-1] if len(macd) >= signal else macd[-1]
    macd_value = macd[-1]
    histogram = macd_value - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(closes, period=BOLLINGER_PERIOD, std_dev=BOLLINGER_STD):
    if len(closes) < period:
        return None, None, None
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return sma, upper_band, lower_band

def calculate_adx(highs, lows, closes, period=ADX_PERIOD):
    if len(highs) < period + 1:
        return None
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(1, len(highs))]
    plus_dm = [highs[i] - highs[i-1] if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0 for i in range(1, len(highs))]
    minus_dm = [lows[i-1] - lows[i] if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0 for i in range(1, len(lows))]
    atr = np.mean(tr[-period:])
    plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr != 0 else 0
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr != 0 else 0
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0
    return dx

def is_doji(open_price, high_price, low_price, close_price):
    body = abs(close_price - open_price)
    range_candle = high_price - low_price
    return body <= 0.1 * range_candle if range_candle != 0 else False

# --- Hàm lấy dữ liệu ---
async def get_historical_data():
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='1m', limit=200)
            if len(ohlcv) < LSTM_WINDOW + 10:
                logger.warning(f"Dữ liệu lịch sử không đủ (lần {attempt + 1}/3)")
                await asyncio.sleep(10)
                continue
            closes = np.array([x[4] for x in ohlcv[-LSTM_WINDOW:]])
            volumes = np.array([x[5] for x in ohlcv[-LSTM_WINDOW:]])
            avg_volume = np.mean(volumes[-10:])
            if avg_volume < 1000:
                logger.warning(f"Thanh khoản thấp (khối lượng trung bình {avg_volume:.2f} < 1000), tạm dừng giao dịch")
                await asyncio.sleep(60)
                continue
            highs = np.array([x[2] for x in ohlcv[-LSTM_WINDOW:]])
            lows = np.array([x[3] for x in ohlcv[-LSTM_WINDOW:]])
            atr = np.array([max(h - l, abs(h - closes[i-1]), abs(l - closes[i-1])) if i > 0 else h - l 
                            for i, (h, l) in enumerate(zip(highs, lows))])[-LSTM_WINDOW:]
            historical_closes = np.array([x[4] for x in ohlcv])
            historical_volumes = np.array([x[5] for x in ohlcv])
            historical_highs = np.array([x[2] for x in ohlcv])
            historical_lows = np.array([x[3] for x in ohlcv])
            return closes, volumes, atr, (historical_closes, historical_volumes, historical_highs, historical_lows, ohlcv)
        except Exception as e:
            logger.error(f"Lỗi lấy dữ liệu lịch sử (lần {attempt + 1}/3): {e}")
            await asyncio.sleep(10)
    return None, None, None, None

async def get_price():
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        return float(ticker['last'])
    except Exception as e:
        logger.error(f"Lỗi lấy giá {SYMBOL}: {e}")
        return None

async def get_balance():
    try:
        balance = exchange.fetch_balance()
        return float(balance['total']['USDT'])
    except Exception as e:
        logger.error(f"Lỗi lấy số dư: {e}")
        return 0

# --- Hàm lấy dữ liệu đa khung thời gian ---
async def get_historical_data_multi_timeframe(timeframe, limit):
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
            if len(ohlcv) < limit:
                logger.warning(f"Dữ liệu {timeframe} không đủ (lần {attempt + 1}/3), chỉ nhận được {len(ohlcv)}/{limit} nến")
                await asyncio.sleep(10)
                continue
            return np.array([x[4] for x in ohlcv])
        except Exception as e:
            logger.error(f"Lỗi lấy dữ liệu khung {timeframe} (lần {attempt + 1}/3): {str(e)}")
            await asyncio.sleep(10)
    logger.error(f"Không thể lấy dữ liệu {timeframe} sau 3 lần thử")
    return None

async def get_trend_confirmation():
    ohlcv_1h = await get_historical_data_multi_timeframe('1h', 50)
    if ohlcv_1h is None:
        return 'sideways'
    closes_1h = ohlcv_1h
    ema_short_1h = np.mean(closes_1h[-10:])
    ema_long_1h = np.mean(closes_1h[-50:])
    return 'up' if ema_short_1h > ema_long_1h else 'down' if ema_short_1h < ema_long_1h else 'sideways'

# --- Hàm dự đoán giá và xác suất thắng ---
async def predict_price_and_confidence(closes, volumes, atr, historical_closes, historical_highs, historical_lows, buy_score, sell_score):
    global scaler, rf_classifier
    if closes is None or len(closes) != LSTM_WINDOW:
        logger.warning("Dữ liệu không đủ để dự đoán giá")
        return None, None, None

    try:
        ema_short = np.array([np.mean(historical_closes[max(0, i-5):i+1]) for i in range(len(historical_closes)-LSTM_WINDOW, len(historical_closes))])
        ema_long = np.array([np.mean(historical_closes[max(0, i-15):i+1]) for i in range(len(historical_closes)-LSTM_WINDOW, len(historical_closes))])
        rsi = np.array([calculate_rsi(historical_closes[:i+1]) or 50 for i in range(len(historical_closes)-LSTM_WINDOW, len(historical_closes))])
        macd_result, _, _ = calculate_macd(historical_closes[-LSTM_WINDOW:]) or (np.zeros(LSTM_WINDOW), 0, 0)
        macd = macd_result[-LSTM_WINDOW:] if isinstance(macd_result, np.ndarray) else np.zeros(LSTM_WINDOW)
        stochastic_result = calculate_stochastic(historical_highs[-LSTM_WINDOW:], historical_lows[-LSTM_WINDOW:], historical_closes[-LSTM_WINDOW:])
        stochastic = np.pad(stochastic_result, (LSTM_WINDOW - len(stochastic_result), 0), mode='edge')[-LSTM_WINDOW:] if stochastic_result is not None else np.full(LSTM_WINDOW, 50)
        ichimoku_result = calculate_ichimoku(historical_highs[-LSTM_WINDOW:], historical_lows[-LSTM_WINDOW:], historical_closes[-LSTM_WINDOW:])
        ichimoku = np.pad(ichimoku_result, (LSTM_WINDOW - len(ichimoku_result), 0), mode='edge')[-LSTM_WINDOW:] if ichimoku_result is not None else np.zeros(LSTM_WINDOW)

        data = np.column_stack((closes, volumes, ema_short, ema_long, atr, rsi, macd, stochastic, ichimoku))
        logger.info(f"Giá đóng cuối cùng trước chuẩn hóa: {closes[-1]:.4f}")
        data_scaled = scaler.transform(data)
        logger.info(f"Giá đóng chuẩn hóa: {data_scaled[-1, 0]:.4f}")
        X = data_scaled.reshape((1, LSTM_WINDOW, 9))
        predicted_scaled = lstm_model.predict(X, verbose=0)
        logger.info(f"Giá dự đoán chuẩn hóa: {predicted_scaled[0][0]:.4f}")
        predicted_price = scaler.inverse_transform(np.array([[predicted_scaled[0][0], data_scaled[-1, 1], 
                                                              data_scaled[-1, 2], data_scaled[-1, 3], 
                                                              data_scaled[-1, 4], data_scaled[-1, 5], 
                                                              data_scaled[-1, 6], data_scaled[-1, 7], 
                                                              data_scaled[-1, 8]]]))[0, 0]
        logger.info(f"Giá dự đoán sau giải mã: {predicted_price:.4f}")

        current_price = closes[-1]
        predicted_change = predicted_price - current_price
        error = abs(predicted_change) / current_price
        if error > 0.1:
            logger.warning(f"Sai số dự đoán {error:.4f} quá lớn (>10%), bỏ qua dự đoán")
            return None, None, None

        confidence_buy = 0.5
        confidence_sell = 0.5
        if rf_classifier is not None and hasattr(rf_classifier, 'estimators_'):
            with sqlite3.connect(DB_PATH) as conn:
                trades = conn.execute("SELECT profit FROM predictions WHERE profit IS NOT NULL").fetchall()
                if len(trades) > 10:
                    probs = rf_classifier.predict_proba([[buy_score, sell_score]])
                    confidence_buy = probs[0][1]
                    confidence_sell = probs[0][0]

        logger.info(f"Dự đoán giá: {predicted_price:.4f}, Confidence Buy: {confidence_buy:.2f}, Confidence Sell: {confidence_sell:.2f}")
        return predicted_price, confidence_buy, confidence_sell
    except Exception as e:
        logger.error(f"Lỗi dự đoán giá: {e}")
        return None, None, None

# --- Hàm xác nhận tín hiệu trước khi vào lệnh ---
async def confirm_trade_signal(buy_score, sell_score, predicted_change, trend, ema_short, ema_long, macd, signal_line, rsi, adx):
    signals = [
        ema_short > ema_long if buy_score > sell_score else ema_short < ema_long,
        predicted_change > 0 if buy_score > sell_score else predicted_change < 0,
        macd[-1] > signal_line if buy_score > sell_score else macd[-1] < signal_line,
        trend == 'up' if buy_score > sell_score else trend == 'down',
        rsi < RSI_OVERBOUGHT if buy_score > sell_score else rsi > RSI_OVERSOLD,
        adx > ADX_SIDWAYS_THRESHOLD
    ]
    confirmation_score = sum(signals) / len(signals)
    logger.info(f"Xác nhận tín hiệu: {confirmation_score:.2f} ({sum(signals)}/{len(signals)} tín hiệu đồng thuận)")
    return confirmation_score >= 0.75

# --- Hàm giao dịch ---
async def place_order_with_tp_sl(side, price, quantity, volatility, predicted_price, atr):
    global position, last_pnl_check_time
    try:
        if position is not None:
            logger.info("Đã có vị thế đang mở, chờ đóng trước khi mở mới")
            return None
        order = exchange.create_order(symbol=SYMBOL, type='market', side=side, amount=quantity, params={'positionSide': 'BOTH'})
        entry_price = float(order['price']) if order.get('price') else price
        logger.info(f"Mở vị thế {side.upper()} thành công: Giá={entry_price}, Số lượng={quantity}")
        last_pnl_check_time = time.time()
        atr_multiplier = atr[-1] / entry_price if atr[-1] != 0 else 0
        take_profit_percent = TAKE_PROFIT_PERCENT * (1 + atr_multiplier)
        stop_loss_percent = STOP_LOSS_PERCENT * (1 + atr_multiplier)
        if side.lower() == 'buy':
            take_profit_price = entry_price * (1 + take_profit_percent)
            stop_loss_price = entry_price * (1 - stop_loss_percent)
            tp_side = 'sell'
            sl_side = 'sell'
        else:
            take_profit_price = entry_price * (1 - take_profit_percent)
            stop_loss_price = entry_price * (1 + stop_loss_percent)
            tp_side = 'buy'
            sl_side = 'buy'
        tp_order = exchange.create_order(
            symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_side, amount=quantity,
            params={'stopPrice': take_profit_price, 'positionSide': 'BOTH', 'reduceOnly': True}
        )
        logger.info(f"Đặt TP: Giá={take_profit_price}, Side={tp_side}, Order ID={tp_order['id']}")
        sl_order = exchange.create_order(
            symbol=SYMBOL, type='STOP_MARKET', side=sl_side, amount=quantity,
            params={'stopPrice': stop_loss_price, 'positionSide': 'BOTH', 'reduceOnly': True}
        )
        logger.info(f"Đặt SL: Giá={stop_loss_price}, Side={sl_side}, Order ID={sl_order['id']}")
        await bot.send_message(chat_id=CHAT_ID, text=f"{side.upper()} {quantity} {SYMBOL} tại {entry_price}, TP={take_profit_price}, SL={stop_loss_price}")
        position = {
            'side': side, 'entry_price': entry_price, 'quantity': quantity,
            'tp_order_id': tp_order['id'], 'sl_order_id': sl_order['id'],
            'tp_price': take_profit_price, 'sl_price': stop_loss_price,
            'open_time': time.time()
        }
        return order
    except Exception as e:
        logger.error(f"Lỗi đặt lệnh {side}: {e}")
        return None

# --- Hàm kiểm tra và đóng vị thế ---
async def check_and_close_position(current_price):
    global position
    if position:
        if position['side'].lower() == 'buy':
            if current_price >= position['tp_price']:
                logger.info(f"Chốt lời thủ công: Giá={current_price} >= TP={position['tp_price']}")
                await close_position('sell', position['quantity'], current_price, "Take Profit")
            elif current_price <= position['sl_price']:
                logger.info(f"Cắt lỗ thủ công: Giá={current_price} <= SL={position['sl_price']}")
                await close_position('sell', position['quantity'], current_price, "Stop Loss")
        else:
            if current_price <= position['tp_price']:
                logger.info(f"Chốt lời thủ công: Giá={current_price} <= TP={position['tp_price']}")
                await close_position('buy', position['quantity'], current_price, "Take Profit")
            elif current_price >= position['sl_price']:
                logger.info(f"Cắt lỗ thủ công: Giá={current_price} >= SL={position['sl_price']}")
                await close_position('buy', position['quantity'], current_price, "Stop Loss")

async def close_position(side, quantity, close_price, close_reason):
    global position
    try:
        order = exchange.create_order(symbol=SYMBOL, type='market', side=side, amount=quantity, params={'positionSide': 'BOTH'})
        gross_profit = (close_price - position['entry_price']) * quantity * LEVERAGE / position['entry_price'] if position['side'].lower() == 'buy' else (position['entry_price'] - close_price) * quantity * LEVERAGE / position['entry_price']
        fee = abs(gross_profit) * TRADING_FEE_PERCENT
        net_profit = gross_profit - fee
        logger.info(f"Đóng vị thế {side.upper()}: Số lượng={quantity}, Giá={order.get('price', close_price)}, Lý do={close_reason}, Lợi nhuận ròng={net_profit:.2f}, Phí={fee:.4f}")
        await bot.send_message(chat_id=CHAT_ID, text=f"Đã đóng {side.upper()} {quantity} {SYMBOL} tại {close_price:.4f}. Lý do: {close_reason}. Lợi nhuận: {net_profit:.2f} USDT")
        position = None
    except Exception as e:
        logger.error(f"Lỗi đóng vị thế: {e}")

# --- Hàm kiểm tra trạng thái vị thế ---
async def check_position_status(current_price):
    global position, last_pnl_check_time
    if position:
        try:
            positions_on_exchange = exchange.fetch_positions([SYMBOL])
            current_position = next((p for p in positions_on_exchange if p['symbol'] == SYMBOL), None)
            if current_position and float(current_position['info']['positionAmt']) == 0:
                profit = (current_position['entryPrice'] - current_position['markPrice']) * position['quantity'] * LEVERAGE / position['entry_price'] if position['side'].lower() == 'buy' else (current_position['markPrice'] - current_position['entryPrice']) * position['quantity'] * LEVERAGE / position['entry_price']
                logger.info("Vị thế đã được đóng trên Binance, đồng bộ với code")
                await bot.send_message(chat_id=CHAT_ID, text=f"Đã đóng {position['side'].upper()} {position['quantity']} {SYMBOL} tại {current_price:.4f}. Lý do: Đóng trên Binance. Lợi nhuận: {profit:.2f} USDT")
                position = None
            else:
                if position['side'].lower() == 'buy':
                    profit = (current_price - position['entry_price']) * position['quantity'] * LEVERAGE / position['entry_price']
                    if profit > 0:
                        trailing_stop_price = current_price * (1 - TRAILING_STOP_PERCENT)
                        if current_price <= trailing_stop_price and position['sl_price'] < trailing_stop_price:
                            logger.info(f"Trailing Stop: Giá={current_price} <= {trailing_stop_price}")
                            await close_position('sell', position['quantity'], current_price, "Trailing Stop")
                else:
                    profit = (position['entry_price'] - current_price) * position['quantity'] * LEVERAGE / position['entry_price']
                    if profit > 0:
                        trailing_stop_price = current_price * (1 + TRAILING_STOP_PERCENT)
                        if current_price >= trailing_stop_price and position['sl_price'] > trailing_stop_price:
                            logger.info(f"Trailing Stop: Giá={current_price} >= {trailing_stop_price}")
                            await close_position('buy', position['quantity'], current_price, "Trailing Stop")
                if ENABLE_PNL_CHECK and time.time() - position['open_time'] >= PNL_CHECK_INTERVAL:
                    unrealized_pnl = float(current_position['unrealizedProfit']) if current_position else 0
                    logger.info(f"PNL kiểm tra: {unrealized_pnl:.2f} USD")
                    if unrealized_pnl >= PNL_THRESHOLD:
                        logger.info(f"PNL dương {unrealized_pnl:.2f} >= {PNL_THRESHOLD} USD, Take Profit thủ công")
                        await close_position('sell' if position['side'].lower() == 'buy' else 'buy', position['quantity'], current_price, "PNL Take Profit")
                        last_pnl_check_time = time.time()
                await check_and_close_position(current_price)
        except Exception as e:
            logger.error(f"Lỗi kiểm tra trạng thái vị thế: {e}")

# --- Hàm lưu dữ liệu ---
async def save_prediction(timestamp, side, predicted_price, entry_price, quantity, actual_price=None, profit=None, buy_score=0, sell_score=0):
    global daily_trades, performance
    try:
        with sqlite3.connect(DB_PATH) as conn:
            predicted_change = predicted_price - entry_price
            actual_change = (actual_price - entry_price) if actual_price else None
            position_value = quantity * entry_price * LEVERAGE
            profit = (actual_price - entry_price) * position_value / entry_price if actual_price else profit
            conn.execute("INSERT INTO predictions (timestamp, predicted_price, entry_price, actual_price, "
                         "predicted_change, actual_change, profit, strategy, buy_score, sell_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (timestamp, predicted_price, entry_price, actual_price, predicted_change, actual_change, profit, "scalping", buy_score, sell_score))
            conn.commit()
            if actual_price:
                daily_trades += 1
                performance['profit'] += profit if profit else 0
                if profit > 0:
                    performance['total_profit'] += profit
                    performance['consecutive_losses'] = 0
                else:
                    performance['total_loss'] += abs(profit)
                    performance['consecutive_losses'] += 1
                trades = conn.execute("SELECT profit FROM predictions WHERE profit IS NOT NULL").fetchall()
                wins = sum(1 for t in trades if t[0] > 0)
                performance['win_rate'] = wins / len(trades) if trades else 0
                await save_trade_log(timestamp, side, entry_price, quantity, profit)
    except Exception as e:
        logger.error(f"Lỗi lưu dự đoán: {e}")

async def save_trade_log(timestamp, side, price, quantity, profit=None):
    try:
        with open('trade_log.csv', 'a') as f:
            f.write(f"{timestamp},{side},{price},{quantity},{profit if profit is not None else ''}\n")
    except Exception as e:
        logger.error(f"Lỗi lưu log giao dịch: {e}")

def calculate_profit_factor():
    total_profit = performance['total_profit']
    total_loss = performance['total_loss']
    return total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 1

# --- Bot chính ---
async def optimized_trading_bot():
    global lstm_model, rf_classifier, is_trading, position, data_buffer, last_retrain_time, last_check_time, last_pnl_check_time, scaler

    model_file = 'lstm_model.keras'
    scaler_file = 'scaler.pkl'
    rf_file = 'rf_classifier.pkl'
    if os.path.exists(model_file):
        try:
            logger.info("Tải mô hình LSTM từ file")
            lstm_model = load_model(model_file)
            if lstm_model.input_shape != (None, LSTM_WINDOW, 9):
                logger.info("Mô hình cũ không tương thích, tạo mới")
                lstm_model = create_lstm_model()
        except Exception as e:
            logger.warning(f"Không thể tải mô hình LSTM: {e}, tạo mới")
            lstm_model = create_lstm_model()
    else:
        lstm_model = create_lstm_model()

    if os.path.exists(scaler_file):
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    else:
        logger.info("Scaler chưa tồn tại, sẽ được tạo khi huấn luyện mô hình")

    if os.path.exists(rf_file):
        with open(rf_file, 'rb') as f:
            rf_classifier = pickle.load(f)
    else:
        rf_classifier = create_rf_classifier()

    historical_data = await get_historical_data()
    if historical_data is None or len(historical_data) != 4:
        logger.error("Không thể lấy dữ liệu lịch sử, thoát")
        return
    closes, volumes, atr, (historical_closes, historical_volumes, historical_highs, historical_lows, ohlcv) = historical_data

    init_db()

    if not os.path.exists(model_file):
        await train_advanced_model(ohlcv, historical_closes, historical_highs, historical_lows, initial=True)
    else:
        await train_advanced_model(ohlcv, historical_closes, historical_highs, historical_lows, initial=False)

    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
    except Exception as e:
        logger.error(f"Lỗi đặt đòn bẩy: {e}")
        return

    last_price = await get_price()
    if not last_price:
        logger.error("Không thể lấy giá ban đầu, thoát")
        return

    while True:
        current_price = await get_price()
        if not current_price:
            await asyncio.sleep(10)
            continue

        balance = await get_balance()
        if performance['profit'] < -DAILY_LOSS_LIMIT or daily_trades >= MAX_DAILY_TRADES or performance['consecutive_losses'] >= 3:
            logger.warning(f"Bot dừng: Profit={performance['profit']:.2f}, Trades={daily_trades}, Profit Factor={calculate_profit_factor():.2f}, Losses={performance['consecutive_losses']}")
            break

        historical_data = await get_historical_data()
        if historical_data is None or len(historical_data) != 4:
            logger.warning("Dữ liệu lịch sử không khả dụng, tạm dừng 10 giây")
            await asyncio.sleep(10)
            continue
        closes, volumes, atr, (historical_closes, historical_volumes, historical_highs, historical_lows, ohlcv) = historical_data

        data_buffer.extend(ohlcv)
        if len(data_buffer) > BUFFER_SIZE:
            data_buffer = data_buffer[-BUFFER_SIZE:]

        current_time = time.time()
        if current_time - last_retrain_time >= RETRAIN_INTERVAL and len(data_buffer) >= LSTM_WINDOW + 10:
            logger.info("Huấn luyện lại mô hình với dữ liệu mới")
            await train_advanced_model(ohlcv, historical_closes, historical_highs, historical_lows, initial=False)
            last_retrain_time = current_time

        ema_short = np.mean(closes[-5:])
        ema_long = np.mean(closes[-15:])
        volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) if np.mean(closes[-10:]) != 0 else 0
        rsi = calculate_rsi(historical_closes) or 50
        macd, signal_line, macd_histogram = calculate_macd(historical_closes) or (np.zeros_like(historical_closes), 0, 0)
        sma, upper_band, lower_band = calculate_bollinger_bands(historical_closes) or (0, 0, 0)
        adx = calculate_adx(historical_highs, historical_lows, historical_closes) or 0
        volume_spike = historical_volumes[-1] > (np.mean(historical_volumes[-10:-1]) * VOLUME_SPIKE_THRESHOLD) if len(historical_volumes) > 10 else False
        doji_pattern = is_doji(ohlcv[-1][1], ohlcv[-1][2], ohlcv[-1][3], ohlcv[-1][4])

        trend = await get_trend_confirmation()

        profit_factor = calculate_profit_factor()

        buy_score = 0
        sell_score = 0
        if ema_short > ema_long:
            buy_score += 20
            logger.info("Điều kiện EMA: Buy +20")
        if ema_short < ema_long:
            sell_score += 20
            logger.info("Điều kiện EMA: Sell +20")
        if profit_factor > 1:
            buy_score += 10
            sell_score += 10
            logger.info("Điều kiện Profit Factor: Buy/Sell +10")
        if current_price > last_price * BREAKOUT_THRESHOLD and current_price > upper_band:
            buy_score += 20
            logger.info("Điều kiện Breakout: Buy +20")
        if current_price < last_price * BREAKDOWN_THRESHOLD and current_price < lower_band:
            sell_score += 20
            logger.info("Điều kiện Breakdown: Sell +20")
        if rsi <= RSI_OVERBOUGHT:
            buy_score += 2
            logger.info("Điều kiện RSI: Buy +2")
        if rsi >= RSI_OVERSOLD:
            sell_score += 2
            logger.info("Điều kiện RSI: Sell +2")
        if volume_spike:
            buy_score += 5
            sell_score += 5
            logger.info("Điều kiện Volume Spike: Buy/Sell +5")
        if macd[-1] > signal_line and macd_histogram > 0:
            buy_score += 15
            logger.info("Điều kiện MACD: Buy +15")
        if macd[-1] < signal_line and macd_histogram < 0:
            sell_score += 15
            logger.info("Điều kiện MACD: Sell +15")
        historical_closes_5m = await get_historical_data_multi_timeframe('5m', 20)
        historical_closes_15m = await get_historical_data_multi_timeframe('15m', 20)
        if historical_closes_5m is not None and historical_closes_15m is not None:
            ema_short_5m = np.mean(historical_closes_5m[-5:])
            ema_long_5m = np.mean(historical_closes_5m[-15:])
            ema_short_15m = np.mean(historical_closes_15m[-5:])
            ema_long_15m = np.mean(historical_closes_15m[-15:])
            if ema_short_5m > ema_long_5m:
                buy_score += 10
                logger.info("Điều kiện EMA 5m: Buy +10")
            if ema_short_5m < ema_long_5m:
                sell_score += 10
                logger.info("Điều kiện EMA 5m: Sell +10")
            if ema_short_15m > ema_long_15m:
                buy_score += 10
                logger.info("Điều kiện EMA 15m: Buy +10")
            if ema_short_15m < ema_long_15m:
                sell_score += 10
                logger.info("Điều kiện EMA 15m: Sell +10")
        if adx > 25:
            buy_score += 5
            sell_score += 5
            logger.info("Điều kiện ADX: Buy/Sell +5")
        if doji_pattern and rsi < 40:
            buy_score += 20
            logger.info("Điều kiện Doji & RSI: Buy +20")
        if doji_pattern and rsi > 60:
            sell_score += 20
            logger.info("Điều kiện Doji & RSI: Sell +20")

        predicted_price, confidence_buy, confidence_sell = await predict_price_and_confidence(
            closes, volumes, atr, historical_closes, historical_highs, historical_lows, buy_score, sell_score
        )
        if predicted_price is None:
            logger.info("Dự đoán giá không hợp lệ, bỏ qua vòng lặp này")
            await asyncio.sleep(10)
            continue

        predicted_change = predicted_price - current_price
        error = abs(predicted_change) / current_price
        if error > 0.1:
            logger.warning(f"Dự đoán giá không đáng tin (sai số {error:.4f} > 10%), giảm điểm Predicted Change")
            if predicted_change > 0:
                buy_score -= 40
            elif predicted_change < 0:
                sell_score -= 40
        else:
            if predicted_change > 0 and abs(predicted_change) > MIN_PREDICTED_CHANGE:
                buy_score += 40
                logger.info("Điều kiện Predicted Change: Buy +40")
            if predicted_change < 0 and abs(predicted_change) > MIN_PREDICTED_CHANGE:
                sell_score += 40
                logger.info("Điều kiện Predicted Change: Sell +40")

        logger.info(f"Price={current_price:.4f}, Predicted={predicted_price:.4f}, EMA ngắn={ema_short:.4f}, "
                    f"EMA dài={ema_long:.4f}, RSI={rsi:.2f}, MACD={macd[-1]:.4f}, ADX={adx:.2f}, Trend={trend}, Win Rate={performance['win_rate']:.2f}")
        logger.info(f"Tổng điểm: Buy Score={buy_score}, Sell Score={sell_score}")

        if position is None and not is_trading:
            timestamp = current_time
            quantity = BASE_AMOUNT * (0.5 if performance['consecutive_losses'] >= 2 else 1.5 if performance['win_rate'] > 0.7 else 1)
            confirmed_buy = await confirm_trade_signal(buy_score, sell_score, predicted_change, trend, ema_short, ema_long, macd, signal_line, rsi, adx)
            if (buy_score >= BUY_THRESHOLD and confidence_buy >= MIN_CONFIDENCE and confirmed_buy and
                trend == 'up' and (predicted_change > 0 or macd[-1] > signal_line)):
                error = abs(predicted_price - current_price) / current_price
                if error > MAX_PREDICTION_ERROR:
                    logger.info(f"Sai số dự đoán {error:.4f} vượt ngưỡng {MAX_PREDICTION_ERROR}, không giao dịch")
                else:
                    logger.info(f"BUY: Score={buy_score}, Confidence={confidence_buy:.2f}, Quantity={quantity}")
                    is_trading = True
                    order = await place_order_with_tp_sl('buy', current_price, quantity, volatility, predicted_price, atr)
                    if order:
                        await save_prediction(timestamp, 'buy', predicted_price, current_price, quantity, buy_score=buy_score, sell_score=sell_score)
                    is_trading = False

            confirmed_sell = await confirm_trade_signal(buy_score, sell_score, predicted_change, trend, ema_short, ema_long, macd, signal_line, rsi, adx)
            if (sell_score >= SELL_THRESHOLD and confidence_sell >= MIN_CONFIDENCE and confirmed_sell and
                trend == 'down' and (predicted_change < 0 or macd[-1] < signal_line)):
                error = abs(predicted_price - current_price) / current_price
                if error > MAX_PREDICTION_ERROR:
                    logger.info(f"Sai số dự đoán {error:.4f} vượt ngưỡng {MAX_PREDICTION_ERROR}, không giao dịch")
                else:
                    logger.info(f"SELL: Score={sell_score}, Confidence={confidence_sell:.2f}, Quantity={quantity}")
                    is_trading = True
                    order = await place_order_with_tp_sl('sell', current_price, quantity, volatility, predicted_price, atr)
                    if order:
                        await save_prediction(timestamp, 'sell', predicted_price, current_price, quantity, buy_score=buy_score, sell_score=sell_score)
                    is_trading = False
            else:
                logger.info(f"No trade: Buy Score={buy_score}, Sell Score={sell_score}")
        else:
            logger.info("Đang có vị thế mở hoặc đang giao dịch, chờ đóng")

        if current_time - last_check_time >= CHECK_INTERVAL and position:
            await check_position_status(current_price)
            last_check_time = current_time

        last_price = current_price
        await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(optimized_trading_bot())
    except KeyboardInterrupt:
        logger.info("Bot dừng bởi người dùng")
    except Exception as e:
        logger.error(f"Lỗi bot: {e}")
