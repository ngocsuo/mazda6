import ccxt.async_support as ccxt 
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
from strategies import TrendFollowing, Scalping, MeanReversion
from colorama import init, Fore, Style
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Khởi tạo colorama
init()

# Cấu hình logging
logger = logging.getLogger('TradingBot')
handler = logging.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(asctime)s [%(log_color)s%(levelname)s%(reset)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    style='%'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)  # Thay vì logging.INFO

# Hàm định dạng log
def format_log_message(message, variables=None, profit=None, section=None):
    formatted = message
    if section:
        if "CHỈ BÁO" in message:
            formatted = f"{Fore.CYAN}{message}{Style.RESET_ALL}"
        elif "THỊ TRƯỜNG" in message:
            formatted = f"{Fore.MAGENTA}{message}{Style.RESET_ALL}"
        elif "CHIẾN LƯỢC" in message:
            formatted = f"{Fore.YELLOW}{message}{Style.RESET_ALL}"
        elif "DỰ ĐOÁN GIÁ" in message:
            formatted = f"{Fore.GREEN}{message}{Style.RESET_ALL}"
        elif "ĐỘ TIN CẬY" in message:
            formatted = f"{Fore.BLUE}{message}{Style.RESET_ALL}"
    if variables:
        for var_name, var_value in variables.items():
            formatted = formatted.replace(
                f"{{{var_name}}}",
                f"{Fore.BLUE}{var_value}{Style.RESET_ALL}"
            )
    if profit is not None:
        profit_str = f"{profit:.2f}"
        color = Fore.GREEN if profit > 0 else Fore.RED
        formatted = formatted.replace(
            f"Profit={profit_str}",
            f"Profit={color}{profit_str}{Style.RESET_ALL}"
        )
    return formatted

def log_with_format(level, message, variables=None, profit=None, section=None):
    if variables is None:
        variables = {}
    formatted_message = format_log_message(message, variables, profit, section)
    if level == 'info':
        logger.info(formatted_message)
    elif level == 'warning':
        logger.warning(formatted_message)
    elif level == 'error':
        logger.error(formatted_message)
    elif level == 'debug':
        logger.debug(formatted_message)

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
BASE_AMOUNT = 30
RISK_PER_TRADE_PERCENT = 0.01
STOP_LOSS_PERCENT = 0.015
TAKE_PROFIT_PERCENT = 0.01
TRADING_FEE_PERCENT = 0.0002
VOLATILITY_RISK_THRESHOLD = 0.02
LSTM_WINDOW = 30
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
RETRAIN_INTERVAL = 120
BUFFER_SIZE = 1000
CHECK_INTERVAL = 15
TRAILING_STOP_PERCENT = 0.01
STOCHASTIC_PERIOD = 14
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SENKO = 52
ENABLE_PNL_CHECK = True
PNL_THRESHOLD = 4.0
PNL_CHECK_INTERVAL = 900
MIN_CONFIDENCE = 0.5
MAX_PREDICTION_ERROR = 0.05
BUY_THRESHOLD = 46
SELL_THRESHOLD = 46
USE_PERCENTAGE = 0.9

# Biến toàn cục
scaler = MinMaxScaler()
lstm_model = None
lstm_classification_model = None  # Thêm mô hình phân loại LSTM
rf_classifier = None
performance = {'profit': 0, 'trades': 0, 'total_profit': 0, 'total_loss': 0, 'consecutive_losses': 0, 'win_rate': 0}
is_trading = False
daily_trades = 0
position = None
data_buffer = []
last_retrain_time = time.time()
last_check_time = time.time()
last_pnl_check_time = time.time()
STRATEGIES = [TrendFollowing(), Scalping(), MeanReversion()]
strategy_performance = {strat.name: {'wins': 0, 'losses': 0} for strat in STRATEGIES}

async def watch_position_and_price():
    global position, current_price
    while True:
        try:
            # Lấy giá bằng fetch_ticker thay vì watch_ticker
            ticker = await exchange.fetch_ticker(SYMBOL)
            current_price = float(ticker['last'])
            log_with_format('debug', "Giá hiện tại từ polling: {price}", 
                           variables={'price': f"{current_price:.2f}"}, section="NET")

            # Kiểm tra và đồng bộ vị thế
            if position:
                positions = await exchange.fetch_positions([SYMBOL])
                current_position = next((p for p in positions if p['symbol'] == SYMBOL), None)
                
                if not current_position or float(current_position['info']['positionAmt']) == 0:
                    log_with_format('info', "Vị thế đã đóng trên sàn, đồng bộ trạng thái", section="MINER")
                    await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Vị thế {position['side']} đã đóng")
                    if position.get('tp_order_id'):
                        await exchange.cancel_order(position['tp_order_id'], SYMBOL)
                    if position.get('sl_order_id'):
                        await exchange.cancel_order(position['sl_order_id'], SYMBOL)
                    position = None
                else:
                    position['entry_price'] = float(current_position['entryPrice'])
                    position['quantity'] = float(current_position['info']['positionAmt'])
                    await update_trailing_stop(current_price, atr=None)  # ATR sẽ được tính trong hàm nếu cần
                    await check_and_close_position(current_price)

            await asyncio.sleep(1)  # Kiểm tra mỗi 1 giây để tránh vượt giới hạn API
        except Exception as e:
            log_with_format('error', "Lỗi polling vị thế/giá: {error}", 
                           variables={'error': str(e)}, section="NET")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Lỗi polling: {str(e)}")
            await asyncio.sleep(5)  # Đợi lâu hơn nếu lỗi để tránh spam API

# --- Hàm khởi tạo mô hình ---
def create_lstm_model():
    log_with_format('debug', "Khởi tạo mô hình LSTM mới")
    inputs = Input(shape=(LSTM_WINDOW, 9))
    x = LSTM(100, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(100)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    log_with_format('info', "Đã tạo thành công mô hình LSTM")
    return model

def create_lstm_classification_model():
    log_with_format('debug', "Khởi tạo mô hình LSTM Classification")
    inputs = Input(shape=(LSTM_WINDOW, 9))
    x = LSTM(100, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(100)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(2, activation='softmax')(x)  # 0: short, 1: long
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    log_with_format('info', "Đã tạo thành công mô hình LSTM Classification")
    return model

def create_rf_classifier():
    log_with_format('debug', "Khởi tạo RandomForest Classifier")
    return RandomForestClassifier(n_estimators=100, random_state=42)

# --- Hàm cơ sở dữ liệu ---
def init_db():
    log_with_format('info', "Khởi tạo cơ sở dữ liệu SQLite")
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
        log_with_format('info', "Cơ sở dữ liệu tại {path} đã sẵn sàng", variables={'path': DB_PATH})
    except Exception as e:
        log_with_format('error', "Lỗi khởi tạo cơ sở dữ liệu: {error}", variables={'error': str(e)})

# --- Hàm huấn luyện mô hình ---
async def train_advanced_model(ohlcv, historical_closes, historical_highs, historical_lows, initial=False):
    global data_buffer, lstm_model, lstm_classification_model, performance, scaler, rf_classifier
    log_with_format('info', "{action} mô hình AI", variables={'action': 'Huấn luyện ban đầu' if initial else 'Cập nhật'}, section="CPU")
    try:
        # Kiểm tra dữ liệu đầu vào
        if len(ohlcv) < LSTM_WINDOW + 10:
            log_with_format('warning', "Dữ liệu không đủ: {current}/{required} mẫu cần thiết",
                           variables={'current': str(len(ohlcv)), 'required': str(LSTM_WINDOW + 10)}, section="CPU")
            return

        # Trích xuất dữ liệu
        closes = np.array([x[4] for x in ohlcv])
        volumes = np.array([x[5] for x in ohlcv])
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])
        atr = np.array([max(h - l, abs(h - closes[i-1]), abs(l - closes[i-1])) if i > 0 else h - l 
                        for i, (h, l) in enumerate(zip(highs, lows))])
        rsi_full = calculate_rsi(closes) or 50
        rsi = np.full(len(closes), rsi_full)
        macd, _, _ = calculate_macd(closes) or (np.zeros(len(closes)), 0, 0)
        stochastic_result = calculate_stochastic(highs, lows, closes)
        stochastic = stochastic_result if stochastic_result is not None else np.full(len(closes), 50)
        ichimoku_result = calculate_ichimoku(highs, lows, closes)
        ichimoku = ichimoku_result if ichimoku_result is not None else np.zeros(len(closes))

        # Đảm bảo độ dài đồng nhất
        min_length = min(len(closes), len(volumes), len(atr), len(rsi), len(macd), len(stochastic), len(ichimoku))
        log_with_format('debug', "Độ dài dữ liệu sau trích xuất: closes={c_len}, volumes={v_len}, min_length={min_len}",
                       variables={'c_len': str(len(closes)), 'v_len': str(len(volumes)), 'min_len': str(min_length)}, section="CPU")

        if min_length < LSTM_WINDOW + 10:
            log_with_format('warning', "Độ dài dữ liệu sau trích xuất không đủ: {min_len}/{required} mẫu cần thiết",
                           variables={'min_len': str(min_length), 'required': str(LSTM_WINDOW + 10)}, section="CPU")
            return

        closes = closes[-min_length:]
        volumes = volumes[-min_length:]
        atr = atr[-min_length:]
        rsi = rsi[-min_length:]
        macd = macd[-min_length:]
        stochastic = stochastic[-min_length:]
        ichimoku = ichimoku[-min_length:]

        # Chuẩn bị dữ liệu cho huấn luyện
        X_reg, y_reg = [], []
        X_cls, y_cls = [], []
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
            if i == LSTM_WINDOW and not initial:
                scaler.fit(data)
                with open('scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
            scaled_data = scaler.transform(data)
            X_reg.append(scaled_data)
            y_reg.append(closes[i])
            X_cls.append(scaled_data)
            y_cls.append(1 if closes[i] > closes[i-1] else 0)

        X_reg = np.array(X_reg) if X_reg else np.array([])
        y_reg = np.array(y_reg) if y_reg else np.array([])
        X_cls = np.array(X_cls) if X_cls else np.array([])
        y_cls = np.array(y_cls) if y_cls else np.array([])

        # Debug dữ liệu
        log_with_format('debug', "Kích thước dữ liệu huấn luyện: X_reg={x_shape}, y_reg={y_shape}, X_cls={xcls_shape}, y_cls={ycls_shape}",
                       variables={'x_shape': str(X_reg.shape), 'y_shape': str(y_reg.shape), 'xcls_shape': str(X_cls.shape), 'ycls_shape': str(y_cls.shape)}, section="CPU")

        # Kiểm tra số lượng mẫu
        if len(X_reg) < 2:
            log_with_format('warning', "Không đủ mẫu dữ liệu để huấn luyện: {samples} mẫu, bỏ qua huấn luyện",
                           variables={'samples': str(len(X_reg))}, section="CPU")
            return

        # Điều chỉnh validation_split nếu dữ liệu ít
        validation_split = 0.1 if len(X_reg) >= 10 else 0.0
        epochs = 50 if initial else 20

        # Huấn luyện mô hình regression
        lstm_model.fit(X_reg, y_reg, epochs=epochs, batch_size=32, validation_split=validation_split, verbose=0)
        lstm_model.save('lstm_model.keras')
        log_with_format('info', "Đã lưu mô hình LSTM Regression", section="CPU")

        # Huấn luyện mô hình classification
        if len(X_cls) > 0:
            lstm_classification_model.fit(X_cls, y_cls, epochs=epochs, batch_size=32, validation_split=validation_split, verbose=0)
            lstm_classification_model.save('lstm_classification_model.keras')
            log_with_format('info', "Đã lưu mô hình LSTM Classification", section="CPU")
        else:
            log_with_format('warning', "Không đủ dữ liệu để huấn luyện mô hình classification", section="CPU")

        # Cập nhật RandomForest Classifier
        with sqlite3.connect(DB_PATH) as conn:
            trades = conn.execute("SELECT buy_score, sell_score, profit FROM predictions WHERE profit IS NOT NULL").fetchall()
            if len(trades) > 10:
                X_rf = np.array([[t[0], t[1]] for t in trades])
                y_rf = np.array([1 if t[2] > 0 else 0 for t in trades])
                rf_classifier.fit(X_rf, y_rf)
                with open('rf_classifier.pkl', 'wb') as f:
                    pickle.dump(rf_classifier, f)
                log_with_format('info', "Đã lưu RandomForest Classifier", section="CPU")

            # Cập nhật performance
            trades = conn.execute("SELECT profit FROM predictions WHERE profit IS NOT NULL").fetchall()
            wins = sum(1 for t in trades if t[0] > 0)
            performance['win_rate'] = wins / len(trades) if trades else 0
            performance['total_profit'] = sum(t[0] for t in trades if t[0] > 0) if trades else 0
            performance['total_loss'] = sum(abs(t[0]) for t in trades if t[0] < 0) if trades else 0
            log_with_format('info', "Hoàn tất huấn luyện. Win Rate: {win_rate}, Total Profit: {profit}, Total Loss: {loss}",
                           variables={'win_rate': f"{performance['win_rate']:.2%}", 'profit': f"{performance['total_profit']:.2f}", 'loss': f"{performance['total_loss']:.2f}"}, section="CPU")

    except Exception as e:
        log_with_format('error', "Lỗi huấn luyện mô hình: {error}", variables={'error': str(e)}, section="CPU")

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

def calculate_vwap(ohlcv):
    if not ohlcv or len(ohlcv) < 1:
        return None
    typical_prices = [(x[2] + x[3] + x[4]) / 3 for x in ohlcv]
    volumes = [x[5] for x in ohlcv]
    cumulative_price_volume = sum(p * v for p, v in zip(typical_prices, volumes))
    cumulative_volume = sum(volumes)
    return cumulative_price_volume / cumulative_volume if cumulative_volume != 0 else None

def calculate_stochastic_rsi(closes, period=14, smooth_k=3, smooth_d=3):
    log_with_format('debug', "Tính Stochastic RSI với closes: {shape}", variables={'shape': str(closes.shape)})
    if len(closes) < period + 1:
        log_with_format('warning', "Dữ liệu không đủ để tính Stochastic RSI: {current}/{required}",
                       variables={'current': str(len(closes)), 'required': str(period + 1)})
        return None, None
    
def is_doji(open_price, high_price, low_price, close_price):
    body = abs(close_price - open_price)
    range_candle = high_price - low_price
    return body <= 0.1 * range_candle if range_candle != 0 else False

def detect_candle_patterns(ohlcv_120s):
    if len(ohlcv_120s) < 2:
        return None
    last_candle = ohlcv_120s[-1]
    prev_candle = ohlcv_120s[-2]
    o, h, l, c = last_candle[1], last_candle[2], last_candle[3], last_candle[4]
    po, ph, pl, pc = prev_candle[1], prev_candle[2], prev_candle[3], prev_candle[4]

    # Bullish Engulfing
    if pc < po and c > o and c > po and o < pc:
        return "bullish_engulfing"
    # Bearish Engulfing
    elif pc > po and c < o and c < po and o > pc:
        return "bearish_engulfing"
    # Doji
    elif is_doji(o, h, l, c):
        return "doji"
    return None

# --- Hàm lấy dữ liệu ---
async def get_historical_data():
    for attempt in range(5):  # Tăng số lần thử từ 3 lên 5
        try:
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, timeframe='1m', limit=2000)
            log_with_format('debug', "Số lượng nến lấy được: {count}", variables={'count': str(len(ohlcv))}, section="NET")
            if len(ohlcv) < LSTM_WINDOW + 10:
                log_with_format('warning', "Dữ liệu OHLCV không đủ: {current}/{required} mẫu, thử lại sau 15s",
                               variables={'current': str(len(ohlcv)), 'required': str(LSTM_WINDOW + 10)}, section="NET")
                await asyncio.sleep(5)  # Tăng thời gian chờ từ 10s lên 15s
                continue
            closes = np.array([x[4] for x in ohlcv[-LSTM_WINDOW:]])
            volumes = np.array([x[5] for x in ohlcv[-LSTM_WINDOW:]])
            avg_volume = np.mean(volumes[-10:]) if len(volumes) > 10 else 0
            if avg_volume < 100:
                log_with_format('warning', "Khối lượng trung bình quá thấp: {avg}, chờ 60s",
                               variables={'avg': f"{avg_volume:.2f}"}, section="NET")
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
            log_with_format('debug', "Dữ liệu lịch sử: closes={c_shape}, volumes={v_shape}",
                           variables={'c_shape': str(historical_closes.shape), 'v_shape': str(historical_volumes.shape)}, section="NET")
            return closes, volumes, atr, (historical_closes, historical_volumes, historical_highs, historical_lows, ohlcv)
        except Exception as e:
            log_with_format('error', "Lỗi lấy dữ liệu (lần {attempt}/5): {error}",
                           variables={'attempt': str(attempt + 1), 'error': str(e)}, section="NET")
            await asyncio.sleep(15)
    log_with_format('error', "Không thể lấy dữ liệu sau 5 lần thử", section="NET")
    return None, None, None, None

async def get_price():
    for attempt in range(5):
        try:
            ticker = await exchange.fetch_ticker(SYMBOL)
            last_price = float(ticker.get('last'))
            if not (1000 <= last_price <= 10000):
                await asyncio.sleep(10)
                continue
            bid = float(ticker.get('bid')) if ticker.get('bid') else None
            ask = float(ticker.get('ask')) if ticker.get('ask') else None
            if bid and ask:
                spread = ask - bid
                spread_percent = (spread / last_price) * 100 if last_price != 0 else float('inf')
                if spread_percent > 0.5:
                    await asyncio.sleep(10)
                    continue
            return last_price
        except Exception as e:
            await asyncio.sleep(5)
    return None

async def get_balance():
    try:
        balance_info = await exchange.fetch_balance({'type': 'future'})
        return float(balance_info['info']['availableBalance'])
    except Exception as e:
        return 0

async def get_historical_data_multi_timeframe(timeframe, limit):
    for attempt in range(3):
        try:
            ohlcv = await exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
            if len(ohlcv) < limit:
                await asyncio.sleep(10)
                continue
            return np.array([x[4] for x in ohlcv]), ohlcv
        except Exception as e:
            await asyncio.sleep(10)
    return None, None

async def get_trend_confirmation():
    ohlcv_1h, _ = await get_historical_data_multi_timeframe('1h', 50)
    ohlcv_4h, _ = await get_historical_data_multi_timeframe('4h', 50)
    trend = 'sideways'
    if ohlcv_1h is not None and ohlcv_4h is not None:
        closes_1h = ohlcv_1h
        closes_4h = ohlcv_4h
        ema_short_1h = np.mean(closes_1h[-10:])
        ema_long_1h = np.mean(closes_1h[-50:])
        ema_short_4h = np.mean(closes_4h[-10:])
        ema_long_4h = np.mean(closes_4h[-50:])
        trend_1h = 'up' if ema_short_1h > ema_long_1h else 'down' if ema_short_1h < ema_long_1h else 'sideways'
        trend_4h = 'up' if ema_short_4h > ema_long_4h else 'down' if ema_short_4h < ema_long_4h else 'sideways'
        trend = trend_4h if trend_4h != 'sideways' else trend_1h
    log_with_format('debug', "Xu hướng xác nhận: Trend={trend}", variables={'trend': trend}, section="THỊ TRƯỜNG")
    return trend

# --- Hàm dự đoán giá và xác suất thắng ---
async def predict_price_and_confidence(closes, volumes, atr, historical_closes, historical_highs, historical_lows, historical_volumes, buy_score, sell_score):
    global scaler, rf_classifier, lstm_classification_model
    log_with_format('info', "=== BẮT ĐẦU DỰ ĐOÁN GIÁ VÀ ĐỘ TIN CẬY ===", section="DỰ ĐOÁN GIÁ")

    if closes is None or len(closes) != LSTM_WINDOW:
        log_with_format('warning', "Dữ liệu không đủ để dự đoán: {current}/{required} mẫu",
                       variables={'current': str(len(closes) if closes else 0), 'required': str(LSTM_WINDOW)}, section="DỰ ĐOÁN GIÁ")
        return None, 0.5, 0.5

    if np.any(np.isnan(closes)) or np.any(np.isinf(closes)) or np.any(closes <= 0):
        log_with_format('warning', "Dữ liệu giá chứa NaN, Inf hoặc giá trị không hợp lệ, bỏ qua dự đoán", section="DỰ ĐOÁN GIÁ")
        return None, 0.5, 0.5

    try:
        if len(historical_closes) < LSTM_WINDOW or len(historical_volumes) < LSTM_WINDOW:
            log_with_format('warning', "Dữ liệu lịch sử không đủ: closes={c_len}, volumes={v_len}/{required} mẫu",
                           variables={'c_len': str(len(historical_closes)), 'v_len': str(len(historical_volumes)), 'required': str(LSTM_WINDOW)}, section="DỰ ĐOÁN GIÁ")
            return None, 0.5, 0.5

        closes_window = historical_closes[-LSTM_WINDOW:]
        volumes_window = historical_volumes[-LSTM_WINDOW:] if len(historical_volumes) >= LSTM_WINDOW else np.full(LSTM_WINDOW, np.mean(historical_volumes[-10:]) if historical_volumes.size > 0 else 0)
        atr_window = atr[-LSTM_WINDOW:] if len(atr) >= LSTM_WINDOW else np.full(LSTM_WINDOW, np.mean(atr[-10:]) if atr.size > 0 else 0)
        highs_window = historical_highs[-LSTM_WINDOW:]
        lows_window = historical_lows[-LSTM_WINDOW:]

        ema_short = np.array([np.mean(historical_closes[max(0, i-5):i+1]) for i in range(len(historical_closes)-LSTM_WINDOW, len(historical_closes))])[-LSTM_WINDOW:]
        ema_long = np.array([np.mean(historical_closes[max(0, i-15):i+1]) for i in range(len(historical_closes)-LSTM_WINDOW, len(historical_closes))])[-LSTM_WINDOW:]
        rsi_full = calculate_rsi(historical_closes[-LSTM_WINDOW:]) or 50
        rsi = np.full(LSTM_WINDOW, rsi_full)
        macd_result, _, _ = calculate_macd(historical_closes[-LSTM_WINDOW:]) or (np.zeros(LSTM_WINDOW), 0, 0)
        macd = macd_result[-LSTM_WINDOW:] if isinstance(macd_result, np.ndarray) and len(macd_result) >= LSTM_WINDOW else np.zeros(LSTM_WINDOW)
        stochastic_result = calculate_stochastic(highs_window, lows_window, closes_window)
        stochastic = np.pad(stochastic_result, (LSTM_WINDOW - len(stochastic_result), 0), mode='edge')[-LSTM_WINDOW:] if stochastic_result is not None else np.full(LSTM_WINDOW, 50)
        ichimoku_result = calculate_ichimoku(highs_window, lows_window, closes_window)
        ichimoku = np.pad(ichimoku_result[-LSTM_WINDOW:], (max(0, LSTM_WINDOW - len(ichimoku_result)), 0), mode='edge')[-LSTM_WINDOW:] if ichimoku_result is not None else np.zeros(LSTM_WINDOW)

        data = np.column_stack((
            closes_window,
            volumes_window,
            ema_short,
            ema_long,
            atr_window,
            rsi,
            macd,
            stochastic,
            ichimoku
        ))
        log_with_format('debug', "Kích thước dữ liệu đầu vào: {shape}", variables={'shape': str(data.shape)}, section="DỰ ĐOÁN GIÁ")

        scaled_data = scaler.transform(data)
        log_with_format('debug', "Kích thước dữ liệu scaled: {shape}", variables={'shape': str(scaled_data.shape)}, section="DỰ ĐOÁN GIÁ")

        X = scaled_data
        y = closes_window
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)
        predicted_price = rf_regressor.predict(X_test[-1].reshape(1, -1))[0]

        current_price = closes[-1]
        predicted_change = predicted_price - current_price
        error = abs(predicted_change) / current_price
        log_with_format('info', "DỰ ĐOÁN GIÁ (Random Forest Regressor): Giá hiện tại={current} | Giá dự đoán={predicted} | Thay đổi={change} ({percent}%) | Sai số={error}",
                       variables={'current': f"{current_price:.4f}", 'predicted': f"{predicted_price:.4f}", 
                                  'change': f"{predicted_change:.4f}", 'percent': f"{predicted_change/current_price*100:.2f}", 'error': f"{error:.4f}"}, section="DỰ ĐOÁN GIÁ")

        if error > MAX_PREDICTION_ERROR:
            log_with_format('warning', "Sai số dự đoán {error} vượt ngưỡng {threshold}, bỏ qua",
                           variables={'error': f"{error:.4f}", 'threshold': f"{MAX_PREDICTION_ERROR}"}, section="DỰ ĐOÁN GIÁ")
            return None, 0.5, 0.5

        confidence_buy = 0.5
        confidence_sell = 0.5
        if lstm_classification_model is not None:
            input_data = scaled_data.reshape(1, LSTM_WINDOW, 9)
            log_with_format('debug', "Kích thước input_data cho LSTM Classification: {shape}", variables={'shape': str(input_data.shape)}, section="DỰ ĐOÁN GIÁ")
            probs = lstm_classification_model.predict(input_data, verbose=0)
            confidence_buy = probs[0][1]  # Xác suất long
            confidence_sell = probs[0][0]  # Xác suất short
            log_with_format('info', "DỰ ĐOÁN LONG/SHORT (LSTM Classification): Buy={buy} | Sell={sell}",
                           variables={'buy': f"{confidence_buy:.2%}", 'sell': f"{confidence_sell:.2%}"}, section="DỰ ĐOÁN GIÁ")

        if rf_classifier is not None and hasattr(rf_classifier, 'estimators_'):
            with sqlite3.connect(DB_PATH) as conn:
                trades = conn.execute("SELECT buy_score, sell_score, profit FROM predictions WHERE profit IS NOT NULL").fetchall()
                if len(trades) > 10:
                    X_rf = np.array([[t[0], t[1]] for t in trades])
                    y_rf = np.array([1 if t[2] > 0 else 0 for t in trades])
                    rf_classifier.fit(X_rf, y_rf)
                    probs_rf = rf_classifier.predict_proba([[buy_score, sell_score]])
                    confidence_buy = (confidence_buy + probs_rf[0][1]) / 2
                    confidence_sell = (confidence_sell + probs_rf[0][0]) / 2
                    log_with_format('info', "ĐỘ TIN CẬY (Kết hợp): Buy={buy} | Sell={sell}",
                                   variables={'buy': f"{confidence_buy:.2%}", 'sell': f"{confidence_sell:.2%}"}, section="ĐỘ TIN CẬY")

        log_with_format('info', "=== KẾT THÚC DỰ ĐOÁN ===", section="DỰ ĐOÁN GIÁ")
        return predicted_price, confidence_buy, confidence_sell
    except Exception as e:
        log_with_format('error', "Lỗi trong quá trình dự đoán: {error}", variables={'error': str(e)}, section="DỰ ĐOÁN GIÁ")
        return None, 0.5, 0.5

# --- Hàm xác nhận tín hiệu ---
async def confirm_trade_signal(buy_score, sell_score, predicted_change, trend, ema_short, ema_long, macd, signal_line, rsi, adx, volume_spike, candle_pattern, stoch_k, stoch_d):
    signals = [
        ema_short > ema_long if buy_score > sell_score else ema_short < ema_long,
        predicted_change > 0 if buy_score > sell_score else predicted_change < 0,
        macd[-1] > signal_line if buy_score > sell_score else macd[-1] < signal_line,
        trend == 'up' if buy_score > sell_score else trend == 'down',
        rsi < RSI_OVERBOUGHT if buy_score > sell_score else rsi > RSI_OVERSOLD,
        adx > 25,
        volume_spike,
        candle_pattern in ["bullish_engulfing"] if buy_score > sell_score else candle_pattern in ["bearish_engulfing"],
        stoch_k > stoch_d and stoch_k < 80 if buy_score > sell_score else stoch_k < stoch_d and stoch_k > 20
    ]
    confirmation_score = sum(signals) / len(signals)
    log_with_format('debug', "Xác nhận tín hiệu: Score={score}, Signals={signals}",
                    variables={'score': f"{confirmation_score:.2f}", 'signals': str(signals)}, section="MINER")
    return confirmation_score >= 0.5

# --- Hàm giao dịch ---
async def place_order_with_tp_sl(side, price, quantity, volatility, predicted_price, atr):
    global position, last_pnl_check_time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            balance_info = await exchange.fetch_balance({'type': 'future'})
            available_balance = float(balance_info['info']['availableBalance'])
            notional_value = price * quantity
            initial_margin = notional_value / LEVERAGE + notional_value * TRADING_FEE_PERCENT
            if available_balance < initial_margin:
                log_with_format('warning', "Không đủ số dư khả dụng để mở vị thế", section="MINER")
                return None

            order = await exchange.create_order(symbol=SYMBOL, type='market', side=side, amount=quantity, params={'positionSide': 'BOTH'})
            entry_price = float(order['price']) if order.get('price') else price
            last_pnl_check_time = time.time()

            atr_multiplier = atr[-1] / entry_price if atr[-1] != 0 else 0
            volatility_adjustment = volatility * 2
            take_profit_percent = TAKE_PROFIT_PERCENT * (1 + atr_multiplier + volatility_adjustment)
            stop_loss_percent = STOP_LOSS_PERCENT * (1 + atr_multiplier + volatility_adjustment)
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

            tp_order = await exchange.create_order(
                symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_side, amount=quantity,
                params={'stopPrice': take_profit_price, 'positionSide': 'BOTH', 'reduceOnly': True}
            )
            sl_order = await exchange.create_order(
                symbol=SYMBOL, type='STOP_MARKET', side=sl_side, amount=quantity,
                params={'stopPrice': stop_loss_price, 'positionSide': 'BOTH', 'reduceOnly': True}
            )

            position = {
                'side': side, 'entry_price': entry_price, 'quantity': quantity,
                'tp_order_id': tp_order['id'], 'sl_order_id': sl_order['id'],
                'tp_price': take_profit_price, 'sl_price': stop_loss_price,
                'open_time': time.time(), 'peak_price': entry_price, 'trough_price': entry_price
            }
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Mở vị thế {side.upper()}: Giá={entry_price:.2f}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, Số lượng={quantity:.2f}")
            return order
        except Exception as e:
            log_with_format('error', "Lỗi đặt lệnh (lần {attempt}/{max}): {error}",
                            variables={'attempt': str(attempt + 1), 'max': str(max_retries), 'error': str(e)}, section="MINER")
            if attempt == max_retries - 1:
                await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Không thể đặt lệnh sau {max_retries} lần thử: {str(e)}")
            await asyncio.sleep(5)
    return None

async def check_and_close_position(current_price):
    global position
    if not position:
        return
    
    side = position['side'].lower()
    entry_price = position['entry_price']
    tp_price = position['tp_price']
    sl_price = position['sl_price']
    quantity = position['quantity']

    try:
        # Kiểm tra Take Profit và Stop Loss
        if side == 'buy':
            if current_price >= tp_price:
                await close_position('sell', quantity, current_price, "Take Profit")
            elif current_price <= sl_price:
                await close_position('sell', quantity, current_price, "Stop Loss")
        elif side == 'sell':
            if current_price <= tp_price:
                await close_position('buy', quantity, current_price, "Take Profit")
            elif current_price >= sl_price:
                await close_position('buy', quantity, current_price, "Stop Loss")
    except Exception as e:
        log_with_format('error', "Lỗi kiểm tra và đóng vị thế: {error}", 
                        variables={'error': str(e)}, section="MINER")
        


async def close_position(side, quantity, close_price, close_reason):
    global position, performance
    try:
        order = await exchange.create_order(symbol=SYMBOL, type='market', side=side, amount=quantity, params={'positionSide': 'BOTH'})
        gross_profit = (close_price - position['entry_price']) * quantity * LEVERAGE / position['entry_price'] if position['side'].lower() == 'buy' else (position['entry_price'] - close_price) * quantity * LEVERAGE / position['entry_price']
        fee = abs(gross_profit) * TRADING_FEE_PERCENT
        net_profit = gross_profit - fee
        
        performance['profit'] += net_profit
        if net_profit > 0:
            performance['total_profit'] += net_profit
            performance['consecutive_losses'] = 0
        else:
            performance['total_loss'] += abs(net_profit)
            performance['consecutive_losses'] += 1
        log_with_format('info', "Đóng vị thế: Lý do={reason}, Profit={profit}",
                        variables={'reason': close_reason}, profit=net_profit, section="MINER")
        
        # Gửi thông báo Telegram khi đóng vị thế
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Đóng vị thế {position['side'].upper()}: Giá={close_price:.2f}, Lý do={close_reason}, Lợi nhuận={net_profit:.2f}")
        
        await save_prediction(time.time(), position['side'], position['tp_price'], position['entry_price'], quantity, close_price, net_profit)
        position = None
    except Exception as e:
        log_with_format('error', "Lỗi đóng vị thế: {error}", variables={'error': str(e)}, section="MINER")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Lỗi đóng vị thế: {str(e)}")
   

async def check_position_status(current_price):
    global position
    if position:
        try:
            positions = await exchange.fetch_positions([SYMBOL])
            current_position = next((p for p in positions if p['symbol'] == SYMBOL), None)
            if not current_position or float(current_position['info']['positionAmt']) == 0:
                log_with_format('info', "Vị thế đã đóng trên sàn, đồng bộ trạng thái", section="MINER")
                if position.get('tp_order_id'):
                    await exchange.cancel_order(position['tp_order_id'], SYMBOL)
                if position.get('sl_order_id'):
                    await exchange.cancel_order(position['sl_order_id'], SYMBOL)
                position = None
            else:
                position['entry_price'] = float(current_position['entryPrice'])
                position['quantity'] = float(current_position['info']['positionAmt'])
                # await update_trailing_stop(current_price)  # Comment để tắt trailing stop
                await check_and_close_position(current_price)
        except Exception as e:
            log_with_format('error', "Lỗi kiểm tra trạng thái vị thế: {error}", 
                           variables={'error': str(e)}, section="MINER")
            

async def update_trailing_stop(current_price, atr=None):
    global position
    if not position:
        return

    try:
        # Nếu không có ATR, lấy từ dữ liệu lịch sử gần nhất
        if atr is None:
            historical_data = await get_historical_data()
            if historical_data:
                atr = historical_data[2][-1]  # Lấy ATR từ dữ liệu lịch sử
            else:
                atr = 0

        atr_multiplier = atr / position['entry_price'] if atr != 0 else 0
        trailing_stop_price = position['sl_price']

        if position['side'].lower() == 'buy':
            peak_price = max(current_price, position.get('peak_price', position['entry_price']))
            position['peak_price'] = peak_price
            new_sl_price = peak_price * (1 - TRAILING_STOP_PERCENT * (1 + atr_multiplier))
            if new_sl_price > trailing_stop_price and new_sl_price > position['entry_price']:
                trailing_stop_price = new_sl_price
        else:
            trough_price = min(current_price, position.get('trough_price', position['entry_price']))
            position['trough_price'] = trough_price
            new_sl_price = trough_price * (1 + TRAILING_STOP_PERCENT * (1 + atr_multiplier))
            if new_sl_price < trailing_stop_price and new_sl_price < position['entry_price']:
                trailing_stop_price = new_sl_price

        if trailing_stop_price != position['sl_price']:
            if position.get('sl_order_id'):
                await exchange.cancel_order(position['sl_order_id'], SYMBOL)
            sl_side = 'sell' if position['side'].lower() == 'buy' else 'buy'
            sl_order = await exchange.create_order(
                symbol=SYMBOL, type='STOP_MARKET', side=sl_side, amount=position['quantity'],
                params={'stopPrice': trailing_stop_price, 'positionSide': 'BOTH', 'reduceOnly': True}
            )
            position['sl_price'] = trailing_stop_price
            position['sl_order_id'] = sl_order['id']
            log_with_format('info', "Cập nhật Trailing Stop: Side={side}, New SL={sl_price}",
                            variables={'side': position['side'], 'sl_price': f"{trailing_stop_price:.2f}"}, section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Cập nhật Trailing Stop {position['side'].upper()}: SL mới={trailing_stop_price:.2f}")
    except Exception as e:
        log_with_format('error', "Lỗi cập nhật Trailing Stop: {error}", 
                        variables={'error': str(e)}, section="MINER")

    
# Cập nhật trong check_position_status
async def check_position_status(current_price):
    global position
    if position:
        try:
            positions_on_exchange = await exchange.fetch_positions([SYMBOL])
            current_position = next((p for p in positions_on_exchange if p['symbol'] == SYMBOL), None)
            
            if not current_position or float(current_position['info']['positionAmt']) == 0:
                log_with_format('info', "Vị thế đã đóng trên sàn, đồng bộ trạng thái cục bộ", section="MINER")
                if position.get('tp_order_id'):
                    await exchange.cancel_order(position['tp_order_id'], SYMBOL)
                if position.get('sl_order_id'):
                    await exchange.cancel_order(position['sl_order_id'], SYMBOL)
                position = None
            else:
                position['entry_price'] = float(current_position['entryPrice'])
                position['quantity'] = float(current_position['info']['positionAmt'])
                await update_trailing_stop(current_price)  # Cập nhật Trailing Stop
                await check_and_close_position(current_price)
        except Exception as e:
            log_with_format('error', "Lỗi kiểm tra trạng thái vị thế: {error}", variables={'error': str(e)}, section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"Lỗi kiểm tra vị thế: {str(e)}")

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
                         (timestamp, predicted_price, entry_price, actual_price, predicted_change, actual_change, profit, "multi_strategy", buy_score, sell_score))
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
    except Exception as e:
        pass

async def save_trade_log(timestamp, side, price, quantity, profit=None):
    try:
        with open('trade_log.csv', 'a') as f:
            f.write(f"{timestamp},{side},{price},{quantity},{profit if profit is not None else ''}\n")
    except Exception as e:
        pass

def calculate_profit_factor():
    total_profit = performance['total_profit']
    total_loss = performance['total_loss']
    return total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 1

def kelly_criterion(win_rate, reward_to_risk):
    kelly = max(0.1, win_rate - (1 - win_rate) / reward_to_risk if reward_to_risk > 0 else 0)
    return kelly

# --- Bot giao dịch chính ---
# Trong hàm optimized_trading_bot, tìm đoạn vòng lặp chính và cập nhật như sau:

# Các hàm hỗ trợ cải tiến
async def check_position_status(current_price):
    global position
    if position:
        try:
            positions_on_exchange = await exchange.fetch_positions([SYMBOL])
            current_position = next((p for p in positions_on_exchange if p['symbol'] == SYMBOL), None)
            
            if not current_position or float(current_position['info']['positionAmt']) == 0:
                log_with_format('info', "Vị thế đã đóng trên sàn, đồng bộ trạng thái cục bộ", section="MINER")
                if position.get('tp_order_id'):
                    await exchange.cancel_order(position['tp_order_id'], SYMBOL)
                if position.get('sl_order_id'):
                    await exchange.cancel_order(position['sl_order_id'], SYMBOL)
                position = None
            else:
                position['entry_price'] = float(current_position['entryPrice'])
                position['quantity'] = float(current_position['info']['positionAmt'])
                await update_trailing_stop(current_price)
                await check_and_close_position(current_price)
        except Exception as e:
            log_with_format('error', "Lỗi kiểm tra trạng thái vị thế: {error}", variables={'error': str(e)}, section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"Lỗi kiểm tra vị thế: {str(e)}")

async def update_trailing_stop(current_price):
    global position
    if not position:
        return

    try:
        trailing_stop_price = position['sl_price']
        if position['side'].lower() == 'buy':
            peak_price = max(current_price, position.get('peak_price', position['entry_price']))
            position['peak_price'] = peak_price
            new_sl_price = peak_price * (1 - TRAILING_STOP_PERCENT)
            if new_sl_price > position['sl_price'] and new_sl_price > position['entry_price']:
                trailing_stop_price = new_sl_price
        else:
            trough_price = min(current_price, position.get('trough_price', position['entry_price']))
            position['trough_price'] = trough_price
            new_sl_price = trough_price * (1 + TRAILING_STOP_PERCENT)
            if new_sl_price < position['sl_price'] and new_sl_price < position['entry_price']:
                trailing_stop_price = new_sl_price

        if trailing_stop_price != position['sl_price']:
            if position.get('sl_order_id'):
                await exchange.cancel_order(position['sl_order_id'], SYMBOL)
            sl_side = 'sell' if position['side'].lower() == 'buy' else 'buy'
            sl_order = await exchange.create_order(
                symbol=SYMBOL, type='STOP_MARKET', side=sl_side, amount=position['quantity'],
                params={'stopPrice': trailing_stop_price, 'positionSide': 'BOTH', 'reduceOnly': True}
            )
            position['sl_price'] = trailing_stop_price
            position['sl_order_id'] = sl_order['id']
            log_with_format('info', "Cập nhật Trailing Stop: Side={side}, New SL={sl_price}",
                            variables={'side': position['side'], 'sl_price': f"{trailing_stop_price:.2f}"}, section="MINER")
    except Exception as e:
        log_with_format('error', "Lỗi cập nhật Trailing Stop: {error}", variables={'error': str(e)}, section="MINER")
        await bot.send_message(chat_id=CHAT_ID, text=f"Lỗi cập nhật Trailing Stop: {str(e)}")

async def close_position(side, quantity, close_price, close_reason):
    global position, performance
    if not position:
        return

    try:
        order = await exchange.create_order(symbol=SYMBOL, type='market', side=side, amount=quantity, 
                                           params={'positionSide': 'BOTH'})
        gross_profit = (close_price - position['entry_price']) * quantity * LEVERAGE / position['entry_price'] \
                      if position['side'].lower() == 'buy' else \
                      (position['entry_price'] - close_price) * quantity * LEVERAGE / position['entry_price']
        fee = abs(gross_profit) * TRADING_FEE_PERCENT
        net_profit = gross_profit - fee

        performance['profit'] += net_profit
        performance['trades'] += 1
        if net_profit > 0:
            performance['total_profit'] += net_profit
            performance['consecutive_losses'] = 0
        else:
            performance['total_loss'] += abs(net_profit)
            performance['consecutive_losses'] += 1
        performance['win_rate'] = performance['total_profit'] / (performance['total_profit'] + performance['total_loss']) \
                                if (performance['total_profit'] + performance['total_loss']) > 0 else 0

        log_with_format('info', "Đóng vị thế: Lý do={reason}, Profit={profit}",
                        variables={'reason': close_reason}, profit=net_profit, section="MINER")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Đóng vị thế {position['side'].upper()}: Giá={close_price:.2f}, "
                                                    f"Lý do={close_reason}, Lợi nhuận={net_profit:.2f}")

        await save_prediction(time.time(), position['side'], position['tp_price'], position['entry_price'], 
                            quantity, close_price, net_profit)
        position = None
    except Exception as e:
        log_with_format('error', "Lỗi đóng vị thế: {error}", variables={'error': str(e)}, section="MINER")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Lỗi đóng vị thế: {str(e)}")
async def start_websocket():
    global position
    while True:
        try:
            # Sử dụng fetch_positions thay vì watch_positions
            positions = await exchange.fetch_positions(symbols=[SYMBOL])
            current_position = next((p for p in positions if p['symbol'] == SYMBOL), None)
            
            if current_position:
                if float(current_position['info']['positionAmt']) == 0 and position:
                    log_with_format('info', "Vị thế đóng qua polling, reset position", section="MINER")
                    await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Vị thế {position['side']} đã đóng")
                    if position.get('tp_order_id'):
                        await exchange.cancel_order(position['tp_order_id'], SYMBOL)
                    if position.get('sl_order_id'):
                        await exchange.cancel_order(position['sl_order_id'], SYMBOL)
                    position = None
                elif position:
                    position['entry_price'] = float(current_position['entryPrice'])
                    position['quantity'] = float(current_position['info']['positionAmt'])
                    current_price = await get_price()
                    if current_price:
                        await update_trailing_stop(current_price)
                        await check_and_close_position(current_price)
            await asyncio.sleep(5)  # Kiểm tra mỗi 5 giây để không quá tải API
        except Exception as e:
            log_with_format('error', "Lỗi polling vị thế: {error}", variables={'error': str(e)}, section="NET")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Lỗi polling vị thế: {str(e)}")
            await asyncio.sleep(5)  # Đợi trước khi thử lại nếu lỗi

def load_historical_performance():
    global performance, strategy_performance, daily_trades
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Tải dữ liệu tổng quan cho performance
            trades = conn.execute("SELECT profit FROM predictions WHERE profit IS NOT NULL").fetchall()
            if trades:
                total_trades = len(trades)
                wins = sum(1 for t in trades if t[0] > 0)
                performance['profit'] = sum(t[0] for t in trades)  # Tổng lợi nhuận/lỗ
                performance['total_profit'] = sum(t[0] for t in trades if t[0] > 0)
                performance['total_loss'] = sum(abs(t[0]) for t in trades if t[0] < 0)
                performance['win_rate'] = wins / total_trades if total_trades > 0 else 0.5
                # Tính số chuỗi thua liên tiếp từ lịch sử (gần đây nhất)
                consecutive_losses = 0
                for trade in reversed(trades):
                    if trade[0] < 0:
                        consecutive_losses += 1
                    else:
                        break
                performance['consecutive_losses'] = consecutive_losses
                log_with_format('info', "Tải hiệu suất lịch sử: Win Rate={win_rate}, Total Profit={profit}, Total Loss={loss}, Consecutive Losses={losses}",
                                variables={'win_rate': f"{performance['win_rate']:.2%}", 
                                           'profit': f"{performance['total_profit']:.2f}", 
                                           'loss': f"{performance['total_loss']:.2f}", 
                                           'losses': str(performance['consecutive_losses'])}, section="CPU")

            # Tải hiệu suất chiến lược (giả sử có cột 'strategy' trong predictions)
            for strat in STRATEGIES:
                strat_trades = conn.execute("SELECT profit FROM predictions WHERE strategy = ? AND profit IS NOT NULL", (strat.name,)).fetchall()
                if strat_trades:
                    wins = sum(1 for t in strat_trades if t[0] > 0)
                    losses = len(strat_trades) - wins
                    strategy_performance[strat.name] = {'wins': wins, 'losses': losses}
                    log_with_format('debug', "Tải lịch sử chiến lược {name}: Wins={wins}, Losses={losses}",
                                    variables={'name': strat.name, 'wins': str(wins), 'losses': str(losses)}, section="CHIẾN LƯỢC")

            # Tính daily_trades từ các giao dịch trong ngày hiện tại
            today_start = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d"))
            daily_trades_count = conn.execute("SELECT COUNT(*) FROM predictions WHERE timestamp >= ? AND profit IS NOT NULL", (today_start,)).fetchone()[0]
            daily_trades = daily_trades_count
            log_with_format('debug', "Số giao dịch hôm nay: {trades}", variables={'trades': str(daily_trades)}, section="CPU")

    except Exception as e:
        log_with_format('error', "Lỗi tải lịch sử hiệu suất: {error}", variables={'error': str(e)}, section="CPU")
        # Giữ giá trị mặc định nếu lỗi
        performance.update({'profit': 0.0, 'win_rate': 0.5, 'consecutive_losses': 0, 'total_profit': 0, 'total_loss': 0})
        strategy_performance.update({strat.name: {'wins': 0, 'losses': 0} for strat in STRATEGIES})
        daily_trades = 0

async def place_order_with_tp_sl(side, price, quantity, volatility, predicted_price, atr):
    global position, last_pnl_check_time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            balance_info = await exchange.fetch_balance({'type': 'future'})
            available_balance = float(balance_info['info']['availableBalance'])
            notional_value = price * quantity
            initial_margin = notional_value / LEVERAGE + notional_value * TRADING_FEE_PERCENT
            if available_balance < initial_margin or position is not None:
                return None

            order = await exchange.create_order(symbol=SYMBOL, type='market', side=side, amount=quantity, params={'positionSide': 'BOTH'})
            entry_price = float(order['price']) if order.get('price') else price
            last_pnl_check_time = time.time()

            atr_multiplier = atr[-1] / entry_price if atr[-1] != 0 else 0
            volatility_adjustment = volatility * 2
            take_profit_percent = TAKE_PROFIT_PERCENT * (1 + atr_multiplier + volatility_adjustment)
            stop_loss_percent = STOP_LOSS_PERCENT * (1 + atr_multiplier + volatility_adjustment)
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

            tp_order = await exchange.create_order(
                symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_side, amount=quantity,
                params={'stopPrice': take_profit_price, 'positionSide': 'BOTH', 'reduceOnly': True}
            )
            sl_order = await exchange.create_order(
                symbol=SYMBOL, type='STOP_MARKET', side=sl_side, amount=quantity,
                params={'stopPrice': stop_loss_price, 'positionSide': 'BOTH', 'reduceOnly': True}
            )

            position = {
                'side': side, 'entry_price': entry_price, 'quantity': quantity,
                'tp_order_id': tp_order['id'], 'sl_order_id': sl_order['id'],
                'tp_price': take_profit_price, 'sl_price': stop_loss_price,
                'open_time': time.time(), 'peak_price': entry_price, 'trough_price': entry_price
            }
            # Gửi thông báo Telegram khi mở vị thế
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Mở vị thế {side.upper()}: Giá={entry_price:.2f}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, Số lượng={quantity:.2f}")
            return order
        except Exception as e:
            log_with_format('error', "Lỗi đặt lệnh (lần {attempt}/{max}): {error}",
                            variables={'attempt': str(attempt + 1), 'max': str(max_retries), 'error': str(e)}, section="MINER")
            if attempt == max_retries - 1:
                await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Không thể đặt lệnh sau {max_retries} lần thử: {str(e)}")
            await asyncio.sleep(5)
    return None

async def watch_price():
    global current_price
    while True:
        try:
            ticker = await exchange.watch_ticker(SYMBOL)
            current_price = float(ticker['last'])
            log_with_format('debug', "Cập nhật giá từ WebSocket: {price}", variables={'price': f"{current_price:.2f}"}, section="NET")
        except Exception as e:
            log_with_format('error', "Lỗi WebSocket giá: {error}", variables={'error': str(e)}, section="NET")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Lỗi WebSocket giá: {str(e)}")
            await asyncio.sleep(5)
            
async def optimized_trading_bot():
    global lstm_model, lstm_classification_model, rf_classifier, is_trading, position, data_buffer, last_retrain_time, last_check_time, last_pnl_check_time, scaler
    global performance, daily_trades, strategy_performance, current_price

    # Khởi tạo
    position = None
    is_trading = False
    last_price = None  # Khai báo last_price
    log_with_format('info', "Bot khởi động, position đã reset", section="CPU")

    # Khởi động task polling
    asyncio.create_task(watch_position_and_price())

    # Vòng lặp chính
    while True:
        current_time = time.time()

        # Chờ giá từ polling
        if not current_price:
            log_with_format('warning', "Chưa có giá từ polling, chờ 2s", section="NET")
            await asyncio.sleep(2)
            continue

        # Cập nhật last_price
        if last_price is None:
            last_price = current_price  # Gán giá trị ban đầu

        # Lấy dữ liệu mới
        historical_data = await get_historical_data()
        if historical_data is None:
            log_with_format('warning', "Không lấy được dữ liệu lịch sử, chờ 10s", section="NET")
            await asyncio.sleep(10)
            continue
        closes, volumes, atr, (historical_closes, historical_volumes, historical_highs, historical_lows, ohlcv) = historical_data
        log_with_format('debug', "Dữ liệu sẵn sàng: closes={c}, volumes={v}", 
                       variables={'c': str(closes.shape), 'v': str(volumes.shape)}, section="NET")

        # Dự đoán giá và độ tin cậy
        try:
            prediction_result = await predict_price_and_confidence(
                closes, volumes, atr, historical_closes, historical_highs, historical_lows, historical_volumes, buy_score=0, sell_score=0
            )
            if prediction_result is None or prediction_result[0] is None:
                log_with_format('warning', "Dự đoán giá thất bại, dùng giá hiện tại", section="DỰ ĐOÁN GIÁ")
                predicted_price = current_price
                confidence_buy = 0.5
                confidence_sell = 0.5
                predicted_change = 0
            else:
                predicted_price, confidence_buy, confidence_sell = prediction_result
                predicted_change = predicted_price - current_price
                log_with_format('info', "Dự đoán: Giá={pred}, Buy Conf={buy}, Sell Conf={sell}, Change={change}",
                               variables={'pred': f"{predicted_price:.2f}", 'buy': f"{confidence_buy:.2%}", 
                                          'sell': f"{confidence_sell:.2%}", 'change': f"{predicted_change:.2f}"}, section="DỰ ĐOÁN GIÁ")
        except Exception as e:
            log_with_format('error', "Lỗi dự đoán giá: {error}", variables={'error': str(e)}, section="DỰ ĐOÁN GIÁ")
            predicted_price = current_price
            confidence_buy = 0.5
            confidence_sell = 0.5
            predicted_change = 0

        # Tính điểm mua/bán
        buy_score = confidence_buy * 100
        sell_score = confidence_sell * 100
        trend = await get_trend_confirmation()
        log_with_format('info', "Điểm: Buy={buy}, Sell={sell}, Trend={trend}",
                       variables={'buy': f"{buy_score:.2f}", 'sell': f"{sell_score:.2f}", 'trend': trend}, section="CHIẾN LƯỢC")

        # Đánh giá chiến lược (xử lý last_price)
        for strategy in STRATEGIES:
            kwargs = {
                'current_price': current_price,
                'last_price': last_price,  # Sử dụng last_price đã khai báo
                'upper_band': calculate_bollinger_bands(historical_closes)[1] or 0,
                'lower_band': calculate_bollinger_bands(historical_closes)[2] or 0,
                'sma': calculate_bollinger_bands(historical_closes)[0] or 0,
                'rsi': calculate_rsi(historical_closes) or 50,
                'ema_short': np.mean(closes[-5:]),
                'ema_long': np.mean(closes[-15:]),
                'adx': calculate_adx(historical_highs, historical_lows, historical_closes) or 0,
                'predicted_change': predicted_change,
                'atr': atr,
                'volume_spike': volumes[-1] > (np.mean(volumes[-10:-1]) * VOLUME_SPIKE_THRESHOLD) if len(volumes) > 10 else False,
                'macd': calculate_macd(historical_closes)[0] or np.zeros_like(closes),
                'signal_line': calculate_macd(historical_closes)[1] or 0,
                'volatility': np.std(closes[-10:]) / np.mean(closes[-10:]) if np.mean(closes[-10:]) != 0 else 0,
                'vwap': calculate_vwap(ohlcv),
                'candle_pattern': detect_candle_patterns(await get_historical_data_multi_timeframe('2m', 5)[1]) if await get_historical_data_multi_timeframe('2m', 5) else None
            }
            if strategy.evaluate_buy(**kwargs):
                buy_score += strategy.weight
            if strategy.evaluate_sell(**kwargs):
                sell_score += strategy.weight

        # Tính khối lượng giao dịch
        balance = await get_balance()
        usable_balance = balance * USE_PERCENTAGE
        notional_value_max = usable_balance * LEVERAGE
        quantity = min(BASE_AMOUNT, notional_value_max / current_price) if current_price != 0 else 0
        log_with_format('debug', "Số dư={bal}, Khối lượng={qty}", 
                       variables={'bal': f"{usable_balance:.2f}", 'qty': f"{quantity:.2f}"}, section="MINER")

        # Thực hiện giao dịch
        if position is not None:
            log_with_format('info', "Đã có vị thế {side}, bỏ qua", 
                           variables={'side': position['side'].upper()}, section="MINER")
        elif not is_trading:
            error = abs(predicted_change) / current_price if predicted_change else 0
            log_with_format('debug', "Điều kiện: Buy={buy}/{thresh}, Sell={sell}/{thresh}, Conf Buy={cb}/{min}, Conf Sell={cs}/{min}, Error={err}/{max}, Trend={trend}",
                           variables={'buy': f"{buy_score:.2f}", 'thresh': str(30), 'sell': f"{sell_score:.2f}", 
                                      'cb': f"{confidence_buy:.2f}", 'cs': f"{confidence_sell:.2f}", 'min': str(0.3), 
                                      'err': f"{error:.4f}", 'max': str(MAX_PREDICTION_ERROR), 'trend': trend}, section="MINER")

            BUY_THRESHOLD_TEMP = 30
            SELL_THRESHOLD_TEMP = 30
            MIN_CONFIDENCE_TEMP = 0.3

            if (buy_score >= BUY_THRESHOLD_TEMP and confidence_buy >= MIN_CONFIDENCE_TEMP and error <= MAX_PREDICTION_ERROR and trend in ['up', 'breakout']):
                is_trading = True
                log_with_format('info', "Đặt lệnh MUA: Giá={price}, Khối lượng={qty}", 
                               variables={'price': f"{current_price:.2f}", 'qty': f"{quantity:.2f}"}, section="MINER")
                order = await place_order_with_tp_sl('buy', current_price, quantity, 0.01, predicted_price, atr)
                if order:
                    await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Đã mở lệnh BUY: Giá={current_price:.2f}, Qty={quantity:.2f}")
                    await save_prediction(time.time(), 'buy', predicted_price, current_price, quantity)
                    daily_trades += 1
                else:
                    log_with_format('error', "Lệnh BUY thất bại", section="MINER")
                is_trading = False
            elif (sell_score >= SELL_THRESHOLD_TEMP and confidence_sell >= MIN_CONFIDENCE_TEMP and error <= MAX_PREDICTION_ERROR and trend in ['down', 'breakout']):
                is_trading = True
                log_with_format('info', "Đặt lệnh BÁN: Giá={price}, Khối lượng={qty}", 
                               variables={'price': f"{current_price:.2f}", 'qty': f"{quantity:.2f}"}, section="MINER")
                order = await place_order_with_tp_sl('sell', current_price, quantity, 0.01, predicted_price, atr)
                if order:
                    await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Đã mở lệnh SELL: Giá={current_price:.2f}, Qty={quantity:.2f}")
                    await save_prediction(time.time(), 'sell', predicted_price, current_price, quantity)
                    daily_trades += 1
                else:
                    log_with_format('error', "Lệnh SELL thất bại", section="MINER")
                is_trading = False

        last_price = current_price  # Cập nhật last_price cho lần lặp sau
        await asyncio.sleep(0.5)# Hàm xử lý dừng bot
async def shutdown_bot(reason, error=None):
    try:
        log_with_format('info' if not error else 'error', f"Bot dừng: {reason}" + (f" - Lỗi: {error}" if error else ""), 
                        variables={'error': str(error)} if error else None, section="NET")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Bot dừng: {reason}" + (f" - Lỗi: {error}" if error else ""))
    except Exception as telegram_error:
        print(f"Lỗi gửi thông báo Telegram khi dừng bot: {telegram_error}")

if __name__ == "__main__":
    try:
        asyncio.run(optimized_trading_bot())
    except KeyboardInterrupt:
        asyncio.run(shutdown_bot("Người dùng dừng bot"))
    except Exception as e:
        asyncio.run(shutdown_bot("Lỗi bot", e))
