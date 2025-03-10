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
from strategies import TrendFollowing, Scalping, MeanReversion
from colorama import init, Fore, Style
from sklearn.model_selection import train_test_split
import RandomForestRegressor 

# Khởi tạo colorama
init()

# Cấu hình logging với màu sắc phong phú
logger = logging.getLogger('TradingBot')
handler = logging.StreamHandler()

# Định dạng log với màu cho các thành phần
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
    secondary_log_colors={},
    style='%'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Hàm định dạng log với màu sắc phong phú
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
BASE_AMOUNT = 40
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
MIN_CONFIDENCE = 0.5  # Giảm từ 0.7 xuống 0.5
MAX_PREDICTION_ERROR = 0.05
BUY_THRESHOLD = 60    # Giảm từ 70
SELL_THRESHOLD = 30
MIN_CONFIDENCE = 0.5

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
STRATEGIES = [TrendFollowing(), Scalping(), MeanReversion()]
strategy_performance = {strat.name: {'wins': 0, 'losses': 0} for strat in STRATEGIES}

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
    global data_buffer, lstm_model, performance, scaler, rf_classifier
    log_with_format('info', "{action} mô hình AI", variables={'action': 'Huấn luyện ban đầu' if initial else 'Cập nhật'})
    try:
        if len(ohlcv) < LSTM_WINDOW + 10:
            log_with_format('warning', "Dữ liệu không đủ: {current}/{required} mẫu cần thiết",
                           variables={'current': str(len(ohlcv)), 'required': str(LSTM_WINDOW + 10)})
            return

        closes = np.array([x[4] for x in ohlcv])
        volumes = np.array([x[5] for x in ohlcv])
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])

        log_with_format('debug', "Chuẩn bị dữ liệu huấn luyện: {samples} mẫu", variables={'samples': str(len(closes))})
        atr = np.array([max(h - l, abs(h - closes[i-1]), abs(l - closes[i-1])) if i > 0 else h - l 
                        for i, (h, l) in enumerate(zip(highs, lows))])
        
        log_with_format('debug', "Tính RSI với {samples} mẫu", variables={'samples': str(len(closes))})
        rsi_full = calculate_rsi(closes) or 50
        rsi = np.full(len(closes), rsi_full)
        
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
            log_with_format('warning', "Không đủ mẫu dữ liệu để huấn luyện: {samples} mẫu", variables={'samples': str(len(X))})
            return

        log_with_format('debug', "Bắt đầu huấn luyện LSTM với {samples} mẫu", variables={'samples': str(len(X))})
        lstm_model.fit(X, y, epochs=20 if not initial else 50, batch_size=32, validation_split=0.1, verbose=0)
        lstm_model.save('lstm_model.keras')
        log_with_format('info', "Đã lưu mô hình LSTM")

        with sqlite3.connect(DB_PATH) as conn:
            trades = conn.execute("SELECT buy_score, sell_score, profit FROM predictions WHERE profit IS NOT NULL").fetchall()
            if len(trades) > 10:
                X_rf = np.array([[t[0], t[1]] for t in trades])
                y_rf = np.array([1 if t[2] > 0 else 0 for t in trades])
                log_with_format('debug', "Huấn luyện RandomForest với {trades} giao dịch", variables={'trades': str(len(trades))})
                rf_classifier.fit(X_rf, y_rf)
                with open('rf_classifier.pkl', 'wb') as f:
                    pickle.dump(rf_classifier, f)
                log_with_format('info', "Đã lưu RandomForest Classifier")

        trades = conn.execute("SELECT profit FROM predictions WHERE profit IS NOT NULL").fetchall()
        wins = sum(1 for t in trades if t[0] > 0)
        performance['win_rate'] = wins / len(trades) if trades else 0
        log_with_format('info', "Hoàn tất huấn luyện. Win Rate: {win_rate}", variables={'win_rate': f"{performance['win_rate']:.2%}"}, section="CPU")
    except Exception as e:
        log_with_format('error', "Lỗi huấn luyện mô hình: {error}", variables={'error': str(e)})

# --- Các hàm chỉ báo kỹ thuật ---
def calculate_stochastic(highs, lows, closes, period=STOCHASTIC_PERIOD):
    log_with_format('debug', "Tính Stochastic với period={period}", variables={'period': str(period)})
    if len(closes) < period + 1:
        log_with_format('warning', "Dữ liệu không đủ để tính Stochastic: {current}/{required}",
                       variables={'current': str(len(closes)), 'required': str(period + 1)})
        return None
    k_values = []
    for i in range(period - 1, len(closes)):
        highest_high = np.max(highs[i-period+1:i+1])
        lowest_low = np.min(lows[i-period+1:i+1])
        k = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low) if (highest_high - lowest_low) != 0 else 50
        k_values.append(k)
    log_with_format('debug', "Stochastic tính được: {value}", variables={'value': f"{k_values[-1]:.2f}"})
    return np.array(k_values)

def calculate_ichimoku(highs, lows, closes):
    log_with_format('debug', "Tính Ichimoku: Tenkan={tenkan}, Kijun={kijun}, Senkou={senkou}",
                   variables={'tenkan': str(ICHIMOKU_TENKAN), 'kijun': str(ICHIMOKU_KIJUN), 'senkou': str(ICHIMOKU_SENKO)})
    if len(closes) < ICHIMOKU_SENKO:
        log_with_format('warning', "Dữ liệu không đủ để tính Ichimoku: {current}/{required}",
                       variables={'current': str(len(closes)), 'required': str(ICHIMOKU_SENKO)})
        return None
    tenkan_sen = np.array([(np.max(highs[i-ICHIMOKU_TENKAN+1:i+1]) + np.min(lows[i-ICHIMOKU_TENKAN+1:i+1])) / 2 
                          for i in range(ICHIMOKU_TENKAN-1, len(closes))])
    kijun_sen = np.array([(np.max(highs[i-ICHIMOKU_KIJUN+1:i+1]) + np.min(lows[i-ICHIMOKU_KIJUN+1:i+1])) / 2 
                          for i in range(ICHIMOKU_KIJUN-1, len(closes))])
    senkou_b = np.array([(np.max(highs[i-ICHIMOKU_SENKO+1:i+1]) + np.min(lows[i-ICHIMOKU_SENKO+1:i+1])) / 2 
                         for i in range(ICHIMOKU_SENKO-1, len(closes))])
    ichimoku = np.column_stack((tenkan_sen[-len(senkou_b):], kijun_sen[-len(senkou_b):], senkou_b)).mean(axis=1)
    log_with_format('debug', "Ichimoku tính được: {value}", variables={'value': f"{ichimoku[-1]:.2f}"})
    return ichimoku if ichimoku.size > 0 else None

def calculate_rsi(closes, period=RSI_PERIOD):
    log_with_format('debug', "Tính RSI với period={period}", variables={'period': str(period)})
    if len(closes) < period + 1:
        log_with_format('warning', "Dữ liệu không đủ để tính RSI: {current}/{required}",
                       variables={'current': str(len(closes)), 'required': str(period + 1)})
        return None
    gains = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        log_with_format('debug', "RSI = 100 do không có losses")
        return 100
    rs = avg_gain / avg_loss
    rsi_value = 100 - (100 / (1 + rs))
    log_with_format('debug', "RSI tính được: {value}", variables={'value': f"{rsi_value:.2f}"})
    return rsi_value

def calculate_macd(closes, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    log_with_format('debug', "Tính MACD: fast={fast}, slow={slow}, signal={signal}",
                   variables={'fast': str(fast), 'slow': str(slow), 'signal': str(signal)})
    if len(closes) < slow + signal:
        log_with_format('warning', "Dữ liệu không đủ để tính MACD: {current}/{required}",
                       variables={'current': str(len(closes)), 'required': str(slow + signal)})
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
    log_with_format('debug', "MACD: {macd}, Signal: {signal}, Histogram: {histogram}",
                   variables={'macd': f"{macd_value:.2f}", 'signal': f"{signal_line:.2f}", 'histogram': f"{histogram:.2f}"})
    return macd, signal_line, histogram

def calculate_bollinger_bands(closes, period=BOLLINGER_PERIOD, std_dev=BOLLINGER_STD):
    log_with_format('debug', "Tính Bollinger Bands: period={period}, std_dev={std_dev}",
                   variables={'period': str(period), 'std_dev': str(std_dev)})
    if len(closes) < period:
        log_with_format('warning', "Dữ liệu không đủ để tính Bollinger: {current}/{required}",
                       variables={'current': str(len(closes)), 'required': str(period)})
        return None, None, None
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    log_with_format('debug', "Bollinger: SMA={sma}, Upper={upper}, Lower={lower}",
                   variables={'sma': f"{sma:.2f}", 'upper': f"{upper_band:.2f}", 'lower': f"{lower_band:.2f}"})
    return sma, upper_band, lower_band

def calculate_adx(highs, lows, closes, period=ADX_PERIOD):
    log_with_format('debug', "Tính ADX với period={period}", variables={'period': str(period)})
    if len(highs) < period + 1:
        log_with_format('warning', "Dữ liệu không đủ để tính ADX: {current}/{required}",
                       variables={'current': str(len(highs)), 'required': str(period + 1)})
        return None
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(1, len(highs))]
    plus_dm = [highs[i] - highs[i-1] if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0 for i in range(1, len(highs))]
    minus_dm = [lows[i-1] - lows[i] if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0 for i in range(1, len(lows))]
    atr = np.mean(tr[-period:])
    plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr != 0 else 0
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr != 0 else 0
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0
    log_with_format('debug', "ADX tính được: {value}", variables={'value': f"{dx:.2f}"})
    return dx

def is_doji(open_price, high_price, low_price, close_price):
    log_with_format('debug', "Kiểm tra nến Doji: Open={open}, High={high}, Low={low}, Close={close}",
                   variables={'open': str(open_price), 'high': str(high_price), 'low': str(low_price), 'close': str(close_price)})
    body = abs(close_price - open_price)
    range_candle = high_price - low_price
    result = body <= 0.1 * range_candle if range_candle != 0 else False
    log_with_format('debug', "Kết quả Doji: {result}", variables={'result': str(result)})
    return result

# --- Hàm lấy dữ liệu ---
async def get_historical_data():
    log_with_format('debug', "Lấy dữ liệu lịch sử cho {symbol}", variables={'symbol': SYMBOL}, section="NET")
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='1m', limit=2000)
            if len(ohlcv) < LSTM_WINDOW + 10:
                log_with_format('warning', "Dữ liệu không đủ (lần {attempt}/3): {current} mẫu",
                               variables={'attempt': str(attempt + 1), 'current': str(len(ohlcv))})
                await asyncio.sleep(10)
                continue
            closes = np.array([x[4] for x in ohlcv[-LSTM_WINDOW:]])
            volumes = np.array([x[5] for x in ohlcv[-LSTM_WINDOW:]])
            avg_volume = np.mean(volumes[-10:])
            log_with_format('debug', "Volume của 10 cây nến gần nhất: {volumes}", variables={'volumes': str(volumes[-10:])}, section="NET")
            if avg_volume < 100:
                log_with_format('warning', "Thanh khoản thấp: Volume trung bình={volume} < 100",
                               variables={'volume': f"{avg_volume:.2f}"})
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
            log_with_format('info', "Đã lấy dữ liệu: {samples} mẫu, Volume TB={volume}",
                           variables={'samples': str(len(ohlcv)), 'volume': f"{avg_volume:.2f}"}, section="NET")
            return closes, volumes, atr, (historical_closes, historical_volumes, historical_highs, historical_lows, ohlcv)
        except Exception as e:
            log_with_format('error', "Lỗi lấy dữ liệu (lần {attempt}/3): {error}",
                           variables={'attempt': str(attempt + 1), 'error': str(e)}, section="NET")
            await asyncio.sleep(10)
    log_with_format('error', "Không thể lấy dữ liệu sau 3 lần thử", section="NET")
    return None, None, None, None

async def get_price():
    log_with_format('info', "Đang lấy giá hiện tại của {symbol}", variables={'symbol': SYMBOL}, section="NET")
    for attempt in range(5):
        try:
            ticker = exchange.fetch_ticker(SYMBOL)
            log_with_format('debug', "Dữ liệu ticker: {ticker}", variables={'ticker': str(ticker)}, section="NET")
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            last_price = ticker.get('last')

            if last_price is None:
                log_with_format('warning', "Không có giá Last, thử lại ({attempt}/5)",
                               variables={'attempt': str(attempt + 1)}, section="NET")
                await asyncio.sleep(10)
                continue

            last_price = float(last_price)
            log_with_format('info', "Giá hiện tại: Last={last}, Bid={bid}, Ask={ask}",
                           variables={'last': f"{last_price:.4f}", 'bid': str(bid) if bid is not None else 'None', 'ask': str(ask) if ask is not None else 'None'}, section="NET")

            if not (1000 <= last_price <= 10000):
                log_with_format('warning', "Giá bất thường: Last={last} không nằm trong khoảng 1000-10000 USDT, thử lại ({attempt}/5)",
                               variables={'last': f"{last_price:.4f}", 'attempt': str(attempt + 1)}, section="NET")
                await asyncio.sleep(10)
                continue

            if bid is not None and ask is not None:
                bid = float(bid)
                ask = float(ask)
                spread = ask - bid
                spread_percent = (spread / last_price) * 100 if last_price != 0 else float('inf')
                log_with_format('info', "Spread: {spread} ({percent}%)",
                               variables={'spread': f"{spread:.4f}", 'percent': f"{spread_percent:.2f}"}, section="NET")
                if spread_percent > 0.5:
                    log_with_format('warning', "Spread lớn: {spread} ({percent}%) vượt ngưỡng 0.5%, thử lại ({attempt}/5)",
                                   variables={'spread': f"{spread:.4f}", 'percent': f"{spread_percent:.2f}", 'attempt': str(attempt + 1)}, section="NET")
                    await asyncio.sleep(10)
                    continue

            return last_price
        except Exception as e:
            log_with_format('error', "Lỗi lấy giá {symbol} (lần {attempt}/5): {error}",
                           variables={'symbol': SYMBOL, 'attempt': str(attempt + 1), 'error': str(e)}, section="NET")
            await asyncio.sleep(10)
    log_with_format('error', "Không thể lấy giá hợp lệ sau 5 lần thử", section="NET")
    return None

async def get_balance():
    log_with_format('debug', "Lấy số dư tài khoản", section="CPU")
    try:
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['total']['USDT'])
        log_with_format('debug', "Số dư USDT: {balance}", variables={'balance': f"{usdt_balance:.2f}"}, section="CPU")
        return usdt_balance
    except Exception as e:
        log_with_format('error', "Lỗi lấy số dư: {error}", variables={'error': str(e)}, section="CPU")
        return 0

async def get_historical_data_multi_timeframe(timeframe, limit):
    log_with_format('debug', "Lấy dữ liệu khung thời gian {timeframe} với limit={limit}",
                   variables={'timeframe': timeframe, 'limit': str(limit)}, section="NET")
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
            if len(ohlcv) < limit:
                log_with_format('warning', "Dữ liệu {timeframe} không đủ (lần {attempt}/3): {current}/{limit}",
                               variables={'timeframe': timeframe, 'attempt': str(attempt + 1), 'current': str(len(ohlcv)), 'limit': str(limit)}, section="NET")
                await asyncio.sleep(10)
                continue
            log_with_format('debug', "Đã lấy dữ liệu {timeframe}: {samples} mẫu",
                           variables={'timeframe': timeframe, 'samples': str(len(ohlcv))}, section="NET")
            return np.array([x[4] for x in ohlcv])
        except Exception as e:
            log_with_format('error', "Lỗi lấy dữ liệu {timeframe} (lần {attempt}/3): {error}",
                           variables={'timeframe': timeframe, 'attempt': str(attempt + 1), 'error': str(e)}, section="NET")
            await asyncio.sleep(10)
    return None

async def get_trend_confirmation():
    log_with_format('debug', "Xác nhận xu hướng từ khung 1h", section="MINER")
    ohlcv_1h = await get_historical_data_multi_timeframe('1h', 50)
    if ohlcv_1h is None:
        log_with_format('warning', "Không xác nhận được xu hướng, mặc định sideways", section="MINER")
        return 'sideways'
    closes_1h = ohlcv_1h
    ema_short_1h = np.mean(closes_1h[-10:])
    ema_long_1h = np.mean(closes_1h[-50:])
    trend = 'up' if ema_short_1h > ema_long_1h else 'down' if ema_short_1h < ema_long_1h else 'sideways'
    log_with_format('debug', "Xu hướng: {trend} (EMA Short={short}, EMA Long={long})",
                   variables={'trend': trend, 'short': f"{ema_short_1h:.2f}", 'long': f"{ema_long_1h:.2f}"}, section="MINER")
    return trend

# --- Hàm dự đoán giá và xác suất thắng ---
async def predict_price_and_confidence(closes, volumes, atr, historical_closes, historical_highs, historical_lows, buy_score, sell_score):
    global scaler, rf_classifier
    log_with_format('info', "=== BẮT ĐẦU DỰ ĐOÁN GIÁ VÀ ĐỘ TIN CẬY ===", section="DỰ ĐOÁN GIÁ")
    
    if closes is None or len(closes) != LSTM_WINDOW:
        log_with_format('warning', "Dữ liệu không đủ để dự đoán: {current}/{required} mẫu",
                       variables={'current': str(len(closes) if closes else 0), 'required': str(LSTM_WINDOW)}, section="DỰ ĐOÁN GIÁ")
        return None, 0.5, 0.5

    if np.any(np.isnan(closes)) or np.any(np.isinf(closes)) or np.any(closes <= 0):
        log_with_format('warning', "Dữ liệu giá chứa NaN, Inf hoặc giá trị không hợp lệ, bỏ qua dự đoán", section="DỰ ĐOÁN GIÁ")
        return None, 0.5, 0.5

    try:
        # Tính các chỉ báo giống như trong train_advanced_model
        ema_short = np.array([np.mean(historical_closes[max(0, i-5):i+1]) for i in range(len(historical_closes)-LSTM_WINDOW, len(historical_closes))])
        ema_long = np.array([np.mean(historical_closes[max(0, i-15):i+1]) for i in range(len(historical_closes)-LSTM_WINDOW, len(historical_closes))])
        rsi_full = calculate_rsi(historical_closes) or 50
        rsi = np.full(LSTM_WINDOW, rsi_full)
        macd_result, _, _ = calculate_macd(historical_closes[-LSTM_WINDOW:]) or (np.zeros(LSTM_WINDOW), 0, 0)
        macd = macd_result[-LSTM_WINDOW:] if isinstance(macd_result, np.ndarray) else np.zeros(LSTM_WINDOW)
        stochastic_result = calculate_stochastic(historical_highs[-LSTM_WINDOW:], historical_lows[-LSTM_WINDOW:], historical_closes[-LSTM_WINDOW:])
        stochastic = np.pad(stochastic_result, (LSTM_WINDOW - len(stochastic_result), 0), mode='edge')[-LSTM_WINDOW:] if stochastic_result is not None else np.full(LSTM_WINDOW, 50)
        ichimoku_result = calculate_ichimoku(historical_highs, historical_lows, historical_closes)
        ichimoku = np.pad(ichimoku_result[-LSTM_WINDOW:], (max(0, LSTM_WINDOW - len(ichimoku_result)), 0), mode='edge')[-LSTM_WINDOW:] if ichimoku_result is not None else np.zeros(LSTM_WINDOW)

        # Tạo đặc trưng với kích thước đồng nhất
        ma5 = np.full(LSTM_WINDOW, np.mean(closes[-5:]) if len(closes) >= 5 else np.mean(closes))  # MA5
        ma20 = np.full(LSTM_WINDOW, np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes))  # MA20
        diff = np.diff(closes)  # Tạo mảng 29 phần tử
        price_change = np.zeros(LSTM_WINDOW)
        price_change[1:] = diff / closes[:-1]  # Phép chia trên 29 phần tử

        # Chuẩn bị dữ liệu với 9 đặc trưng (đồng bộ với train_advanced_model)
        data = np.column_stack((closes, volumes, ema_short, ema_long, atr, rsi, macd, stochastic, ichimoku))
        log_with_format('debug', "Kích thước dữ liệu: {shape}", variables={'shape': str(data.shape)}, section="DỰ ĐOÁN GIÁ")
        scaled_data = scaler.transform(data)

        # Chuẩn bị dữ liệu cho Random Forest
        X = scaled_data  # Sử dụng toàn bộ cửa sổ
        y = closes  # Mục tiêu là giá đóng cửa
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Huấn luyện Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        predicted_price = rf_model.predict(X_test[-1].reshape(1, -1))[0]

        current_price = closes[-1]
        predicted_change = predicted_price - current_price
        error = abs(predicted_change) / current_price
        log_with_format('info', "DỰ ĐOÁN GIÁ (Random Forest): Giá hiện tại={current} | Giá dự đoán={predicted} | Thay đổi={change} ({percent}%) | Sai số={error}",
                       variables={'current': f"{current_price:.4f}", 'predicted': f"{predicted_price:.4f}", 
                                  'change': f"{predicted_change:.4f}", 'percent': f"{predicted_change/current_price*100:.2f}", 'error': f"{error:.4f}"}, section="DỰ ĐOÁN GIÁ")

        if error > MAX_PREDICTION_ERROR:
            log_with_format('warning', "Sai số dự đoán {error} vượt ngưỡng {threshold}, bỏ qua",
                           variables={'error': f"{error:.4f}", 'threshold': f"{MAX_PREDICTION_ERROR}"}, section="DỰ ĐOÁN GIÁ")
            return None, 0.5, 0.5

        # Sử dụng rf_classifier cho độ tin cậy
        confidence_buy = 0.5
        confidence_sell = 0.5
        if rf_classifier is not None and hasattr(rf_classifier, 'estimators_'):
            with sqlite3.connect(DB_PATH) as conn:
                trades = conn.execute("SELECT profit FROM predictions WHERE profit IS NOT NULL").fetchall()
                log_with_format('debug', "Số giao dịch trong predictions: {trades}", variables={'trades': str(len(trades))}, section="ĐỘ TIN CẬY")
                if len(trades) > 10:
                    probs = rf_classifier.predict_proba([[buy_score, sell_score]])
                    confidence_buy = probs[0][1]
                    confidence_sell = probs[0][1]
                    log_with_format('info', "ĐỘ TIN CẬY: Buy={buy} | Sell={sell} (dựa trên {trades} giao dịch)",
                                   variables={'buy': f"{confidence_buy:.2%}", 'sell': f"{confidence_sell:.2%}", 'trades': str(len(trades))}, section="ĐỘ TIN CẬY")
                else:
                    log_with_format('debug', "Chưa đủ dữ liệu lịch sử ({trades}/10), dùng confidence mặc định 50%",
                                   variables={'trades': str(len(trades))}, section="ĐỘ TIN CẬY")

        log_with_format('info', "=== KẾT THÚC DỰ ĐOÁN ===", section="DỰ ĐOÁN GIÁ")
        return predicted_price, confidence_buy, confidence_sell
    except Exception as e:
        log_with_format('error', "Lỗi trong quá trình dự đoán: {error}", variables={'error': str(e)}, section="DỰ ĐOÁN GIÁ")
        return None, 0.5, 0.5
    
# --- Hàm xác nhận tín hiệu ---
async def confirm_trade_signal(buy_score, sell_score, predicted_change, trend, ema_short, ema_long, macd, signal_line, rsi, adx):
    log_with_format('debug', "Xác nhận tín hiệu giao dịch", section="MINER")
    signals = [
        ema_short > ema_long if buy_score > sell_score else ema_short < ema_long,
        predicted_change > 0 if buy_score > sell_score else predicted_change < 0,
        macd[-1] > signal_line if buy_score > sell_score else macd[-1] < signal_line,
        trend == 'up' if buy_score > sell_score else trend == 'down',
        rsi < RSI_OVERBOUGHT if buy_score > sell_score else rsi > RSI_OVERSOLD,
        adx < 20 or adx > 25
    ]
    log_with_format('debug', "Tín hiệu: EMA={ema}, Pred={pred}, MACD={macd}, Trend={trend}, RSI={rsi}, ADX={adx}",
                   variables={'ema': str(signals[0]), 'pred': str(signals[1]), 'macd': str(signals[2]),
                              'trend': str(signals[3]), 'rsi': str(signals[4]), 'adx': str(signals[5])}, section="MINER")
    confirmation_score = sum(signals) / len(signals)
    log_with_format('info', "Xác nhận tín hiệu: Score={score} ({signals}/{total} tín hiệu đồng thuận)",
                   variables={'score': f"{confirmation_score:.2f}", 'signals': str(sum(signals)), 'total': str(len(signals))}, section="MINER")
    return confirmation_score >= 0.33

# --- Hàm giao dịch ---
async def place_order_with_tp_sl(side, price, quantity, volatility, predicted_price, atr):
    global position, last_pnl_check_time
    log_with_format('info', "📈 ĐẶT LỆNH {side} | Giá={price} | Số lượng={quantity}",
                   variables={'side': side.upper(), 'price': f"{price:.4f}", 'quantity': f"{quantity:.2f}"}, section="MINER")
    try:
        if position is not None:
            log_with_format('warning', "Đã có vị thế mở, không thể đặt lệnh mới", section="MINER")
            return None
        order = exchange.create_order(symbol=SYMBOL, type='market', side=side, amount=quantity, params={'positionSide': 'BOTH'})
        entry_price = float(order['price']) if order.get('price') else price
        log_with_format('info', "Thành công: {side} tại {price}", variables={'side': side.upper(), 'price': f"{entry_price:.4f}"}, section="MINER")
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
        sl_order = exchange.create_order(
            symbol=SYMBOL, type='STOP_MARKET', side=sl_side, amount=quantity,
            params={'stopPrice': stop_loss_price, 'positionSide': 'BOTH', 'reduceOnly': True}
        )
        log_with_format('info', "Đã đặt TP={tp} | SL={sl}",
                       variables={'tp': f"{take_profit_price:.4f}", 'sl': f"{stop_loss_price:.4f}"}, section="MINER")
        await bot.send_message(chat_id=CHAT_ID, text=f"📈 {side.upper()} {quantity} {SYMBOL} tại {entry_price:.4f}, TP={take_profit_price:.4f}, SL={stop_loss_price:.4f}")
        position = {
            'side': side, 'entry_price': entry_price, 'quantity': quantity,
            'tp_order_id': tp_order['id'], 'sl_order_id': sl_order['id'],
            'tp_price': take_profit_price, 'sl_price': stop_loss_price,
            'open_time': time.time()
        }
        return order
    except Exception as e:
        log_with_format('error', "Lỗi đặt lệnh {side}: {error}", variables={'side': side, 'error': str(e)}, section="MINER")
        return None

async def check_and_close_position(current_price):
    global position
    log_with_format('debug', "Kiểm tra vị thế: Giá hiện tại={price}", variables={'price': f"{current_price:.4f}"}, section="MINER")
    if position:
        if position['side'].lower() == 'buy':
            if current_price >= position['tp_price']:
                await close_position('sell', position['quantity'], current_price, "Take Profit")
            elif current_price <= position['sl_price']:
                await close_position('sell', position['quantity'], current_price, "Stop Loss")
        else:
            if current_price <= position['tp_price']:
                await close_position('buy', position['quantity'], current_price, "Take Profit")
            elif current_price >= position['sl_price']:
                await close_position('buy', position['quantity'], current_price, "Stop Loss")

async def close_position(side, quantity, close_price, close_reason):
    global position
    log_with_format('info', "📉 ĐÓNG VỊ THẾ {side} | Giá={price} | Lý do={reason}",
                   variables={'side': side.upper(), 'price': f"{close_price:.4f}", 'reason': close_reason}, section="MINER")
    try:
        order = exchange.create_order(symbol=SYMBOL, type='market', side=side, amount=quantity, params={'positionSide': 'BOTH'})
        gross_profit = (close_price - position['entry_price']) * quantity * LEVERAGE / position['entry_price'] if position['side'].lower() == 'buy' else (position['entry_price'] - close_price) * quantity * LEVERAGE / position['entry_price']
        fee = abs(gross_profit) * TRADING_FEE_PERCENT
        net_profit = gross_profit - fee
        log_with_format('info', "Đóng thành công: Lợi nhuận ròng=Profit USDT", profit=net_profit, section="MINER")
        await bot.send_message(chat_id=CHAT_ID, text=f"📉 Đóng {side.upper()} {quantity} {SYMBOL} tại {close_price:.4f}. Lý do: {close_reason}. Lợi nhuận: {net_profit:.2f} USDT")
        position = None
    except Exception as e:
        log_with_format('error', "Lỗi đóng vị thế: {error}", variables={'error': str(e)}, section="MINER")

async def check_position_status(current_price):
    global position, last_pnl_check_time
    log_with_format('debug', "Kiểm tra trạng thái vị thế: Giá={price}", variables={'price': f"{current_price:.4f}"}, section="MINER")
    if position:
        try:
            positions_on_exchange = exchange.fetch_positions([SYMBOL])
            current_position = next((p for p in positions_on_exchange if p['symbol'] == SYMBOL), None)
            if current_position and float(current_position['info']['positionAmt']) == 0:
                profit = float(current_position['unrealizedProfit']) if current_position else 0
                log_with_format('info', "Vị thế đã đóng trên sàn: Lợi nhuận=Profit USDT", profit=profit, section="MINER")
                await bot.send_message(chat_id=CHAT_ID, text=f"📉 Đóng {position['side'].upper()} {position['quantity']} {SYMBOL} tại {current_price:.4f}. Lý do: Đóng trên Binance. Lợi nhuận: {profit:.2f} USDT")
                position = None
            else:
                if position['side'].lower() == 'buy':
                    profit = (current_price - position['entry_price']) * position['quantity'] * LEVERAGE / position['entry_price']
                    if profit > 0:
                        trailing_stop_price = current_price * (1 - TRAILING_STOP_PERCENT)
                        if current_price <= trailing_stop_price and position['sl_price'] < trailing_stop_price:
                            await close_position('sell', position['quantity'], current_price, "Trailing Stop")
                else:
                    profit = (position['entry_price'] - current_price) * position['quantity'] * LEVERAGE / position['entry_price']
                    if profit > 0:
                        trailing_stop_price = current_price * (1 + TRAILING_STOP_PERCENT)
                        if current_price >= trailing_stop_price and position['sl_price'] > trailing_stop_price:
                            await close_position('buy', position['quantity'], current_price, "Trailing Stop")
                if ENABLE_PNL_CHECK and time.time() - position['open_time'] >= PNL_CHECK_INTERVAL:
                    unrealized_pnl = float(current_position['unrealizedProfit']) if current_position else 0
                    if unrealized_pnl >= PNL_THRESHOLD:
                        await close_position('sell' if position['side'].lower() == 'buy' else 'buy', position['quantity'], current_price, "PNL Take Profit")
                        last_pnl_check_time = time.time()
                await check_and_close_position(current_price)
        except Exception as e:
            log_with_format('error', "Lỗi kiểm tra trạng thái vị thế: {error}", variables={'error': str(e)}, section="MINER")

# --- Hàm lưu dữ liệu ---
async def save_prediction(timestamp, side, predicted_price, entry_price, quantity, actual_price=None, profit=None, buy_score=0, sell_score=0):
    global daily_trades, performance
    log_with_format('debug', "Lưu dữ liệu giao dịch: {side} | Entry={entry}",
                   variables={'side': side.upper(), 'entry': f"{entry_price:.4f}"}, section="CPU")
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
                log_with_format('info', "Lưu giao dịch: {side} | Predicted={predicted} | Profit=Profit",
                               variables={'side': side.upper(), 'predicted': f"{predicted_price:.4f}"}, profit=profit, section="CPU")
                await save_trade_log(timestamp, side, entry_price, quantity, profit)
    except Exception as e:
        log_with_format('error', "Lỗi lưu dự đoán: {error}", variables={'error': str(e)}, section="CPU")

async def save_trade_log(timestamp, side, price, quantity, profit=None):
    log_with_format('debug', "Lưu log giao dịch: {side} | Giá={price}",
                   variables={'side': side.upper(), 'price': f"{price:.4f}"}, section="CPU")
    try:
        with open('trade_log.csv', 'a') as f:
            f.write(f"{timestamp},{side},{price},{quantity},{profit if profit is not None else ''}\n")
        log_with_format('debug', "Đã lưu log giao dịch vào trade_log.csv", section="CPU")
    except Exception as e:
        log_with_format('error', "Lỗi lưu log giao dịch: {error}", variables={'error': str(e)}, section="CPU")

def calculate_profit_factor():
    total_profit = performance['total_profit']
    total_loss = performance['total_loss']
    factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 1
    log_with_format('debug', "Profit Factor: {factor}", variables={'factor': f"{factor:.2f}"}, section="CPU")
    return factor

def kelly_criterion(win_rate, reward_to_risk):
    kelly = max(0.1, win_rate - (1 - win_rate) / reward_to_risk if reward_to_risk > 0 else 0)
    log_with_format('debug', "Kelly Criterion: Win Rate={win_rate}, R:R={rr}, Kelly={kelly}",
                   variables={'win_rate': f"{win_rate:.2%}", 'rr': f"{reward_to_risk:.2f}", 'kelly': f"{kelly:.2f}"}, section="CPU")
    return kelly

# --- Bot chính ---
async def optimized_trading_bot():
    global lstm_model, rf_classifier, is_trading, position, data_buffer, last_retrain_time, last_check_time, last_pnl_check_time, scaler

    log_with_format('info', "=== KHỞI ĐỘNG BOT TRADING ===", section="NET")
    log_with_format('info', "Symbol: {symbol} | Leverage: {leverage}x",
                   variables={'symbol': SYMBOL, 'leverage': str(LEVERAGE)}, section="NET")

    model_file = 'lstm_model.keras'
    scaler_file = 'scaler.pkl'
    rf_file = 'rf_classifier.pkl'
    if os.path.exists(model_file):
        try:
            lstm_model = load_model(model_file)
            if lstm_model.input_shape != (None, LSTM_WINDOW, 9):
                log_with_format('info', "Mô hình cũ không tương thích, tạo lại mô hình", section="CPU")
                lstm_model = create_lstm_model()
                os.remove(model_file)
            else:
                log_with_format('info', "Đã tải mô hình LSTM từ file", section="CPU")
        except Exception as e:
            log_with_format('warning', "Không thể tải mô hình LSTM: {error}, tạo mới", variables={'error': str(e)}, section="CPU")
            lstm_model = create_lstm_model()
    else:
        lstm_model = create_lstm_model()

    if os.path.exists(scaler_file):
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
            log_with_format('info', "Đã tải scaler từ file", section="CPU")
    else:
        log_with_format('info', "Scaler chưa tồn tại, sẽ tạo khi huấn luyện", section="CPU")

    if os.path.exists(rf_file):
        with open(rf_file, 'rb') as f:
            rf_classifier = pickle.load(f)
            log_with_format('info', "Đã tải RandomForest từ file", section="CPU")
    else:
        rf_classifier = create_rf_classifier()

    historical_data = await get_historical_data()
    if historical_data is None or len(historical_data) != 4:
        log_with_format('error', "Không thể lấy dữ liệu lịch sử, thoát", section="NET")
        return
    closes, volumes, atr, (historical_closes, historical_volumes, historical_highs, historical_lows, ohlcv) = historical_data

    init_db()
    if not os.path.exists(model_file):
        await train_advanced_model(ohlcv, historical_closes, historical_highs, historical_lows, initial=True)
    else:
        await train_advanced_model(ohlcv, historical_closes, historical_highs, historical_lows, initial=False)

    try:
        exchange.set_leverage(LEVERAGE, SYMBOL)
        log_with_format('info', "Đã đặt đòn bẩy {leverage}x", variables={'leverage': str(LEVERAGE)}, section="NET")
    except Exception as e:
        log_with_format('error', "Lỗi đặt đòn bẩy: {error}", variables={'error': str(e)}, section="NET")
        return

    last_price = await get_price()
    if not last_price:
        log_with_format('error', "Không thể lấy giá ban đầu sau nhiều lần thử, thoát", section="NET")
        return

    while True:
        current_time = time.time()
        current_price = await get_price()
        if not current_price:
            log_with_format('warning', "Không lấy được giá hợp lệ, chờ 15s để thử lại", section="NET")
            await asyncio.sleep(15)
            continue

        balance = await get_balance()
        log_with_format('debug', "Trạng thái tài khoản: Balance={balance} USDT | Profit=Profit | Trades={trades}/{max_trades}",
                       variables={'balance': f"{balance:.2f}", 'trades': str(daily_trades), 'max_trades': str(MAX_DAILY_TRADES)}, profit=performance['profit'], section="CPU")
        if performance['profit'] < -DAILY_LOSS_LIMIT or daily_trades >= MAX_DAILY_TRADES or performance['consecutive_losses'] >= 3:
            log_with_format('warning', "DỪNG BOT: Profit=Profit | Trades={trades} | Losses liên tiếp={losses}",
                           variables={'trades': str(daily_trades), 'losses': str(performance['consecutive_losses'])}, profit=performance['profit'], section="CPU")
            break

        historical_data = await get_historical_data()
        if historical_data is None:
            log_with_format('warning', "Không lấy được dữ liệu, chờ 10s", section="NET")
            await asyncio.sleep(10)
            continue
        closes, volumes, atr, (historical_closes, historical_volumes, historical_highs, historical_lows, ohlcv) = historical_data

        data_buffer.extend(ohlcv)
        if len(data_buffer) > BUFFER_SIZE:
            data_buffer = data_buffer[-BUFFER_SIZE:]

        current_time = time.time()
        if current_time - last_retrain_time >= RETRAIN_INTERVAL and len(data_buffer) >= LSTM_WINDOW + 10:
            log_with_format('info', "--- HUẤN LUYỆN LẠI MÔ HÌNH ---", section="CPU")
            await train_advanced_model(ohlcv, historical_closes, historical_highs, historical_lows, initial=False)
            last_retrain_time = current_time

        ema_short = np.mean(closes[-5:])
        ema_long = np.mean(closes[-15:])
        volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) if np.mean(closes[-10:]) != 0 else 0
        rsi = calculate_rsi(historical_closes) or 50
        macd, signal_line, _ = calculate_macd(historical_closes) or (np.zeros_like(closes), 0, 0)
        sma, upper_band, lower_band = calculate_bollinger_bands(historical_closes) or (0, 0, 0)
        adx = calculate_adx(historical_highs, historical_lows, historical_closes) or 0
        volume_spike = volumes[-1] > (np.mean(volumes[-10:-1]) * VOLUME_SPIKE_THRESHOLD) if len(volumes) > 10 else False
        log_with_format('info', "CHỈ BÁO: EMA Short={short} | EMA Long={long} | Volatility={vol} | RSI={rsi} | ADX={adx}",
                       variables={'short': f"{ema_short:.4f}", 'long': f"{ema_long:.4f}", 'vol': f"{volatility:.4f}", 'rsi': f"{rsi:.2f}", 'adx': f"{adx:.2f}"}, section="CHỈ BÁO")

        prediction_result = await predict_price_and_confidence(
            closes, volumes, atr, historical_closes, historical_highs, historical_lows, buy_score=0, sell_score=0
        )
        if prediction_result is None or prediction_result[0] is None:
            log_with_format('warning', "Không thể dự đoán giá, sử dụng chỉ báo kỹ thuật để giao dịch", section="MINER")
            predicted_price = current_price
            confidence_buy = 0.5
            confidence_sell = 0.5
            predicted_change = 0
        else:
            predicted_price, confidence_buy, confidence_sell = prediction_result
            predicted_change = predicted_price - current_price if predicted_price else 0

        trend = await get_trend_confirmation()
        market_state = 'trending' if adx > 25 else 'sideways' if adx < 20 else 'breakout' if volume_spike else 'normal'
        log_with_format('info', "THỊ TRƯỜNG: State={state} | Trend={trend} | Volume Spike={spike}",
                       variables={'state': market_state, 'trend': trend, 'spike': str(volume_spike)}, section="THỊ TRƯỜNG")

        buy_score = 0
        sell_score = 0
        active_strategies = []
        for strategy in STRATEGIES:
            wins = strategy_performance[strategy.name]['wins']
            losses = strategy_performance[strategy.name]['losses']
            total = wins + losses
            dynamic_weight = strategy.weight * (wins / total if total > 0 else 1.0)

            kwargs = {
                'current_price': current_price,
                'last_price': last_price,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'sma': sma,
                'rsi': rsi,
                'ema_short': ema_short,
                'ema_long': ema_long,
                'adx': adx,
                'predicted_change': predicted_change,
                'atr': atr,
                'volume_spike': volume_spike,
                'macd': macd,
                'signal_line': signal_line,
                'volatility': volatility
            }
            if strategy.name == 'Scalping':
                if ema_short < ema_long and volatility < 0.001 and 50 <= rsi <= 60 and trend == 'down':
                    sell_score += dynamic_weight * 1.5
                    active_strategies.append(strategy.name)
                elif ema_short > ema_long and volatility > 0.0005 and rsi < 50:
                    buy_score += dynamic_weight * 1.5
                    active_strategies.append(strategy.name)
            if strategy.evaluate_buy(**kwargs):
                buy_score += dynamic_weight
                active_strategies.append(strategy.name)
            if strategy.evaluate_sell(**kwargs):
                sell_score += dynamic_weight
                active_strategies.append(strategy.name)

        buy_score += (confidence_buy * 50) if confidence_buy else 0
        sell_score += (confidence_sell * 50) if confidence_sell else 0

        historical_closes_5m = await get_historical_data_multi_timeframe('5m', 20)
        historical_closes_15m = await get_historical_data_multi_timeframe('15m', 20)
        if historical_closes_5m is not None and historical_closes_15m is not None:
            ema_short_5m = np.mean(historical_closes_5m[-5:])
            ema_long_5m = np.mean(historical_closes_5m[-15:])
            ema_short_15m = np.mean(historical_closes_15m[-5:])
            ema_long_15m = np.mean(historical_closes_15m[-15:])
            if ema_short_5m > ema_long_5m and ema_short_15m > ema_long_15m:
                buy_score += 20
            elif ema_short_5m < ema_long_5m and ema_short_15m < ema_long_15m:
                sell_score += 20

        log_with_format('info', "CHIẾN LƯỢC: Buy Score={buy} | Sell Score={sell} | Active={active}",
                       variables={'buy': f"{buy_score:.2f}", 'sell': f"{sell_score:.2f}", 'active': ', '.join(active_strategies)}, section="CHIẾN LƯỢC")

        reward_to_risk = TAKE_PROFIT_PERCENT / STOP_LOSS_PERCENT
        kelly = kelly_criterion(performance['win_rate'], reward_to_risk)
        quantity = BASE_AMOUNT * kelly
        log_with_format('debug', "Kelly Criterion: Win Rate={win_rate} | R:R={rr} | Quantity={quantity}",
                       variables={'win_rate': f"{performance['win_rate']:.2f}", 'rr': f"{reward_to_risk:.2f}", 'quantity': f"{quantity:.4f}"}, section="CPU")

        if position is None and not is_trading:
            timestamp = current_time
            confirmed = await confirm_trade_signal(buy_score, sell_score, predicted_change, trend, ema_short, ema_long, macd, signal_line, rsi, adx)
            error = abs(predicted_change) / current_price if predicted_change else 0
            log_with_format('debug', "Kiểm tra điều kiện: Buy={buy}, Sell={sell}, Confirmed={conf}, Trend={trend}, Error={error}, Conf Buy={conf_buy}, Conf Sell={conf_sell}",
                           variables={'buy': str(buy_score), 'sell': str(sell_score), 'conf': str(confirmed), 'trend': trend, 'error': f"{error:.4f}", 
                                      'conf_buy': str(confidence_buy), 'conf_sell': str(confidence_sell)}, section="MINER")
            if (buy_score >= BUY_THRESHOLD and confidence_buy >= MIN_CONFIDENCE and confirmed and
                error <= MAX_PREDICTION_ERROR and trend in ['up', 'breakout', 'down']):
                log_with_format('info', "📈 TÍN HIỆU MUA: Score={score} | Confidence={conf} | Quantity={quantity}",
                               variables={'score': f"{buy_score:.2f}", 'conf': f"{confidence_buy:.2%}", 'quantity': f"{quantity:.2f}"}, section="MINER")
                is_trading = True
                order = await place_order_with_tp_sl('buy', current_price, quantity, volatility, predicted_price, atr)
                if order:
                    await save_prediction(timestamp, 'buy', predicted_price, current_price, quantity, buy_score=buy_score, sell_score=sell_score)
                    for strat in active_strategies:
                        strategy_performance[strat]['wins' if order['price'] else 'losses'] += 1
                is_trading = False
            elif (sell_score >= SELL_THRESHOLD and confidence_sell >= MIN_CONFIDENCE and confirmed and
                  error <= MAX_PREDICTION_ERROR and trend in ['down', 'breakout']):
                log_with_format('info', "📉 TÍN HIỆU BÁN: Score={score} | Confidence={conf} | Quantity={quantity}",
                               variables={'score': f"{sell_score:.2f}", 'conf': f"{confidence_sell:.2%}", 'quantity': f"{quantity:.2f}"}, section="MINER")
                is_trading = True
                order = await place_order_with_tp_sl('sell', current_price, quantity, volatility, predicted_price, atr)
                if order:
                    await save_prediction(timestamp, 'sell', predicted_price, current_price, quantity, buy_score=buy_score, sell_score=sell_score)
                    for strat in active_strategies:
                        strategy_performance[strat]['wins' if order['price'] else 'losses'] += 1
                is_trading = False

        if current_time - last_check_time >= CHECK_INTERVAL and position:
            await check_position_status(current_price)
            last_check_time = current_time

        last_price = current_price
        await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(optimized_trading_bot())
    except KeyboardInterrupt:
        log_with_format('info', "Bot dừng bởi người dùng", section="NET")
    except Exception as e:
        log_with_format('error', "Lỗi bot: {error}", variables={'error': str(e)}, section="NET")
