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
import traceback
import math
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
STOP_LOSS_PERCENT = 0.03
TAKE_PROFIT_PERCENT = 0.02
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
    last_logged_price = None
    price_change_threshold = 0.5
    log_interval = 60
    last_log_time = time.time()

    while True:
        try:
            ticker = await exchange.fetch_ticker(SYMBOL)
            current_price = float(ticker['last'])

            current_time = time.time()
            if last_logged_price is None or \
               abs(current_price - last_logged_price) >= price_change_threshold or \
               (current_time - last_log_time >= log_interval):
                log_with_format('debug', "Giá hiện tại từ polling: {price}",
                               variables={'price': f"{current_price:.2f}"}, section="NET")
                last_logged_price = current_price
                last_log_time = current_time

            if position:
                positions = await exchange.fetch_positions([SYMBOL])
                current_position = next((p for p in positions if p['symbol'] == SYMBOL), None)
                if not current_position or float(current_position['info']['positionAmt']) == 0:
                    log_with_format('info', "Vị thế đã đóng trên sàn: Side={side}, Qty={qty}",
                                   variables={'side': position['side'], 'qty': str(position['quantity'])}, section="MINER")
                    await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Vị thế {position['side']} đã đóng")
                    position = None
                else:
                    position['entry_price'] = float(current_position['entryPrice'])
                    position['quantity'] = float(current_position['info']['positionAmt'])

                    # Kiểm tra xem SL/TP có thực sự được đặt trên sàn không
                    sl_exists = False
                    tp_exists = False
                    if 'sl_order_id' in position and position['sl_order_id']:
                        try:
                            sl_order = await exchange.fetch_order(position['sl_order_id'], SYMBOL)
                            sl_exists = sl_order['status'] == 'open'
                        except Exception as e:
                            log_with_format('error', "Lỗi kiểm tra trạng thái SL: {error}",
                                           variables={'error': str(e)}, section="MINER")
                    if 'tp_order_id' in position and position['tp_order_id']:
                        try:
                            tp_order = await exchange.fetch_order(position['tp_order_id'], SYMBOL)
                            tp_exists = tp_order['status'] == 'open'
                        except Exception as e:
                            log_with_format('error', "Lỗi kiểm tra trạng thái TP: {error}",
                                           variables={'error': str(e)}, section="MINER")

                    if not (sl_exists and tp_exists):
                        log_with_format('error', "SL/TP không tồn tại trên sàn (SL={sl_status}, TP={tp_status}), đóng vị thế ngay",
                                       variables={'sl_status': str(sl_exists), 'tp_status': str(tp_exists)}, section="MINER")
                        close_side = 'sell' if position['side'] == 'buy' else 'buy'
                        await close_position(close_side, position['quantity'], current_price, "SL/TP Missing")
                        continue

                    await check_and_close_position(current_price)

            await asyncio.sleep(5)
        except Exception as e:
            log_with_format('error', "Lỗi polling vị thế/giá: {error}",
                           variables={'error': str(e)}, section="NET")
            await asyncio.sleep(5)


async def test_order_placement():
    global exchange, current_price, position
    log_with_format('info', "=== BẮT ĐẦU KIỂM TRA ĐẶT VỊ THẾ VÀ TP/SL ===", section="MINER")
    MIN_NOTIONAL_VALUE = 20.0  # Giá trị tối thiểu theo quy định Binance Futures
    TEST_QUANTITY = 1  # Set cứng TEST_QUANTITY = 1
    wait_time = 5
    max_retries = 3
    monitoring_time = 120

    try:
        # Lấy thông tin symbol từ API Binance (vẫn giữ để debug nếu cần)
        markets = await exchange.fetch_markets()
        symbol_info = next((m for m in markets if m['symbol'] == SYMBOL), None)
        if not symbol_info:
            log_with_format('error', "Không tìm thấy thông tin symbol {symbol}", 
                            variables={'symbol': SYMBOL}, section="MINER")
            return False
        precision = symbol_info['precision']['amount']
        quantity_precision = int(precision) if isinstance(precision, (int, float)) else 0
        log_with_format('debug', "Thông tin symbol: Quantity Precision={prec}", 
                        variables={'prec': str(quantity_precision)}, section="MINER")

        # Lấy giá hiện tại
        current_price = await get_price()
        if current_price is None:
            log_with_format('error', "Không thể lấy giá hiện tại để test", section="NET")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: Không lấy được giá hiện tại")
            return False

        # Tính notional_value với TEST_QUANTITY set cứng
        notional_value = TEST_QUANTITY * current_price
        log_with_format('info', "Thông số test: Giá={price}, Số lượng={qty}, Giá trị={notional}",
                        variables={'price': f"{current_price:.2f}", 'qty': str(TEST_QUANTITY), 
                                   'notional': f"{notional_value:.2f}"}, section="MINER")

        if notional_value < MIN_NOTIONAL_VALUE:
            log_with_format('error', "TEST_QUANTITY set cứng không đủ lớn: Notional={notional} < 20", 
                            variables={'notional': f"{notional_value:.2f}"}, section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: Giá trị danh nghĩa {notional_value:.2f} < 20")
            return False

        # Mở vị thế test với lệnh BUY
        log_with_format('info', "Đặt vị thế BUY để test", section="MINER")
        test_order = await place_order_with_tp_sl(
            side='buy',
            price=current_price,
            quantity=TEST_QUANTITY,
            volatility=0.0,
            predicted_price=current_price * 1.02,
            atr=0
        )

        if test_order is None or not position:
            log_with_format('error', "Không thể mở vị thế test BUY", section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: Không mở được vị thế BUY")
            return False

        # Đợi để sàn xử lý
        await asyncio.sleep(wait_time)

        # Kiểm tra trạng thái vị thế và TP/SL
        positions = await exchange.fetch_positions([SYMBOL])
        current_position = next((p for p in positions if p['symbol'] == SYMBOL), None)
        if not current_position or float(current_position['info']['positionAmt']) == 0:
            log_with_format('error', "Vị thế test không tồn tại trên sàn", section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: Vị thế không tồn tại")
            return False

        # Kiểm tra TP/SL trên sàn
        sl_exists = False
        tp_exists = False
        if 'sl_order_id' in position and position['sl_order_id']:
            try:
                sl_order = await exchange.fetch_order(position['sl_order_id'], SYMBOL)
                sl_exists = sl_order['status'] == 'open'
                log_with_format('info', "Trạng thái SL: {status}, Giá SL: {sl_price}",
                                variables={'status': sl_order['status'], 'sl_price': f"{position['sl_price']:.2f}"}, 
                                section="MINER")
            except Exception as e:
                log_with_format('error', "Lỗi kiểm tra SL: {error}", variables={'error': str(e)}, section="MINER")

        if 'tp_order_id' in position and position['tp_order_id']:
            try:
                tp_order = await exchange.fetch_order(position['tp_order_id'], SYMBOL)
                tp_exists = tp_order['status'] == 'open'
                log_with_format('info', "Trạng thái TP: {status}, Giá TP: {tp_price}",
                                variables={'status': tp_order['status'], 'tp_price': f"{position['tp_price']:.2f}"}, 
                                section="MINER")
            except Exception as e:
                log_with_format('error', "Lỗi kiểm tra TP: {error}", variables={'error': str(e)}, section="MINER")

        if not (sl_exists and tp_exists):
            log_with_format('error', "TP/SL không được đặt thành công: SL={sl_status}, TP={tp_status}",
                            variables={'sl_status': str(sl_exists), 'tp_status': str(tp_exists)}, section="MINER")
            await close_position('sell', TEST_QUANTITY, current_price, "Test Failed: TP/SL Missing")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: SL={sl_exists}, TP={tp_exists}")
            return False

        # Theo dõi giá để kiểm tra TP/SL
        log_with_format('info', "Bắt đầu theo dõi giá để kiểm tra TP/SL trong {time}s",
                        variables={'time': str(monitoring_time)}, section="MINER")
        start_time = time.time()
        tp_triggered = False
        sl_triggered = False

        while time.time() - start_time < monitoring_time:
            current_price = await get_price()
            if current_price is None:
                continue

            log_with_format('debug', "Giá hiện tại: {price} | SL={sl} | TP={tp}",
                            variables={'price': f"{current_price:.2f}", 'sl': f"{position['sl_price']:.2f}", 
                                       'tp': f"{position['tp_price']:.2f}"}, section="MINER")

            positions = await exchange.fetch_positions([SYMBOL])
            current_position = next((p for p in positions if p['symbol'] == SYMBOL), None)

            if not current_position or float(current_position['info']['positionAmt']) == 0:
                if 'tp_order_id' in position:
                    tp_order = await exchange.fetch_order(position['tp_order_id'], SYMBOL)
                    if tp_order['status'] in ['closed', 'filled']:
                        tp_triggered = True
                        log_with_format('info', "TP đã kích hoạt tại giá {price}", 
                                        variables={'price': f"{current_price:.2f}"}, section="MINER")
                        break
                if 'sl_order_id' in position:
                    sl_order = await exchange.fetch_order(position['sl_order_id'], SYMBOL)
                    if sl_order['status'] in ['closed', 'filled']:
                        sl_triggered = True
                        log_with_format('info', "SL đã kích hoạt tại giá {price}", 
                                        variables={'price': f"{current_price:.2f}"}, section="MINER")
                        break
                break

            await asyncio.sleep(5)

        # Đánh giá kết quả
        if tp_triggered:
            log_with_format('info', "Kiểm tra thành công: TP đã kích hoạt đúng", section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thành công: TP kích hoạt tại {current_price:.2f}")
            position = None
            return True
        elif sl_triggered:
            log_with_format('info', "Kiểm tra thành công: SL đã kích hoạt đúng", section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thành công: SL kích hoạt tại {current_price:.2f}")
            position = None
            return True
        else:
            log_with_format('warning', "TP/SL không kích hoạt trong thời gian theo dõi, đóng vị thế thủ công", section="MINER")
            for attempt in range(max_retries):
                try:
                    await close_position('sell', TEST_QUANTITY, current_price, "Test Completed: Manual Close")
                    log_with_format('info', "Đã đóng vị thế test thủ công", section="MINER")
                    break
                except Exception as e:
                    log_with_format('error', "Lỗi đóng vị thế test (lần {attempt}/{max}): {error}",
                                    variables={'attempt': str(attempt+1), 'max': str(max_retries), 'error': str(e)},
                                    section="MINER")
                    if attempt == max_retries - 1:
                        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: Không đóng được vị thế test")
                        return False
                    await asyncio.sleep(wait_time)

            log_with_format('info', "Kiểm tra hoàn tất: TP/SL được đặt thành công nhưng không kích hoạt trong {time}s",
                            variables={'time': str(monitoring_time)}, section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test hoàn tất: TP/SL được đặt nhưng không kích hoạt trong {monitoring_time}s")
            return True

    except Exception as e:
        log_with_format('error', "Lỗi nghiêm trọng trong kiểm tra: {error}\nTrace: {trace}", 
                        variables={'error': str(e), 'trace': traceback.format_exc()}, section="MINER")
        if position:
            for attempt in range(max_retries):
                try:
                    await close_position('sell', TEST_QUANTITY, current_price, "Test Failed: Exception")
                    break
                except Exception as close_error:
                    log_with_format('error', "Lỗi đóng vị thế khi test thất bại: {error}", 
                                    variables={'error': str(close_error)}, section="MINER")
                    if attempt == max_retries - 1:
                        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] KHẨN CẤP: Test thất bại và không đóng được vị thế: {str(e)}")

        return False
        
async def check_and_close_position(current_price):
    global position
    if not position:
        return

    entry_price = position['entry_price']
    quantity = position['quantity']
    side = position['side']

    unrealized_pnl = (current_price - entry_price) * quantity * LEVERAGE / entry_price \
                     if side == 'buy' else \
                     (entry_price - current_price) * quantity * LEVERAGE / entry_price

    if unrealized_pnl < -PNL_THRESHOLD:
        log_with_format('warning', "PNL vị thế {side} vượt ngưỡng: {pnl}, đóng vị thế", 
                       variables={'side': side.upper(), 'pnl': f"{unrealized_pnl:.2f}"}, section="MINER")
        await close_position('sell' if side == 'buy' else 'buy', quantity, current_price, "PNL Threshold Exceeded")
    else:
        log_with_format('debug', "PNL vị thế {side}: {pnl}", 
                       variables={'side': side.upper(), 'pnl': f"{unrealized_pnl:.2f}"}, section="MINER")
        
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

async def place_order_with_tp_sl(side, price, quantity, volatility, predicted_price, atr):
    global position
    max_retries = 3
    wait_time = 5  # Thời gian chờ giữa các lần thử

    try:
        # Đặt lệnh MARKET để mở vị thế
        order = None
        for attempt in range(max_retries):
            try:
                order = await exchange.create_order(
                    symbol=SYMBOL,
                    type='MARKET',
                    side=side,
                    amount=quantity,
                    params={'positionSide': 'BOTH'}  # Đảm bảo tương thích với hedge mode
                )
                entry_price = float(order['price']) if order.get('price') else price
                log_with_format('info', "Đã mở vị thế {side}: Giá={price}, Số lượng={qty}",
                               variables={'side': side.upper(), 'price': f"{entry_price:.2f}", 'qty': f"{quantity:.2f}"},
                               section="MINER")
                break
            except Exception as e:
                log_with_format('error', "Lỗi đặt lệnh {side} (lần {attempt}/{max}): {error}",
                               variables={'side': side, 'attempt': str(attempt+1), 'max': str(max_retries), 'error': str(e)},
                               section="MINER")
                if attempt == max_retries - 1:
                    raise Exception("Không thể mở vị thế sau 3 lần thử")
                await asyncio.sleep(wait_time)

        # Đợi để sàn xử lý lệnh mở vị thế
        await asyncio.sleep(wait_time)

        # Lấy giá hiện tại để tính TP/SL chính xác
        current_price = await get_price()
        if current_price is None:
            log_with_format('error', "Không thể lấy giá hiện tại để đặt SL/TP", section="NET")
            raise Exception("Không thể lấy giá hiện tại")

        # Sửa logic tính SL và TP
        STOP_LOSS_PERCENT_ADJUSTED = STOP_LOSS_PERCENT + volatility
        TAKE_PROFIT_PERCENT_ADJUSTED = TAKE_PROFIT_PERCENT + volatility

        if side == 'buy':
            sl_price = entry_price * (1 - STOP_LOSS_PERCENT_ADJUSTED)  # SL dưới giá vào lệnh
            tp_price = entry_price * (1 + TAKE_PROFIT_PERCENT_ADJUSTED)  # TP trên giá vào lệnh
        else:  # side == 'sell'
            sl_price = entry_price * (1 + STOP_LOSS_PERCENT_ADJUSTED)  # SL trên giá vào lệnh
            tp_price = entry_price * (1 - TAKE_PROFIT_PERCENT_ADJUSTED)  # TP dưới giá vào lệnh

        # Điều chỉnh giá để tránh "immediately triggered"
        min_price_diff = entry_price * 0.005  # Khoảng cách tối thiểu 0.5% so với giá hiện tại
        if side == 'buy':
            sl_price = min(sl_price, current_price - min_price_diff)  # SL không được vượt quá giá hiện tại
            tp_price = max(tp_price, current_price + min_price_diff)  # TP không được dưới giá hiện tại
        else:
            sl_price = max(sl_price, current_price + min_price_diff)  # SL không được dưới giá hiện tại
            tp_price = min(tp_price, current_price - min_price_diff)  # TP không được vượt quá giá hiện tại

        # Làm tròn giá theo tick size của Binance
        symbol_info = await exchange.fetch_trading_fees(SYMBOL)
        tick_size = 0.01  # Giả định tick size, cần lấy từ API thực tế
        sl_price = round(sl_price / tick_size) * tick_size
        tp_price = round(tp_price / tick_size) * tick_size

        log_with_format('debug', "Tính toán SL/TP: Giá vào lệnh={entry}, Giá hiện tại={current}, SL={sl}, TP={tp}",
                        variables={'entry': f"{entry_price:.2f}", 'current': f"{current_price:.2f}", 
                                   'sl': f"{sl_price:.2f}", 'tp': f"{tp_price:.2f}"}, section="MINER")

        # Lưu thông tin vị thế tạm thời
        position = {
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'sl_price': sl_price,
            'tp_price': tp_price
        }

        # Đặt Stop Loss
        sl_order_id = None
        for attempt in range(max_retries):
            try:
                sl_order = await exchange.create_order(
                    symbol=SYMBOL,
                    type='STOP_MARKET',
                    side='sell' if side == 'buy' else 'buy',
                    amount=quantity,
                    params={
                        'stopPrice': sl_price,
                        'reduceOnly': True,
                        'positionSide': 'BOTH'
                    }
                )
                sl_order_id = sl_order['id']
                # Kiểm tra trạng thái lệnh SL
                sl_status = await exchange.fetch_order(sl_order_id, SYMBOL)
                if sl_status['status'] == 'open':
                    position['sl_order_id'] = sl_order_id
                    log_with_format('info', "Đã đặt SL cho vị thế {side}: SL={sl}",
                                   variables={'side': side.upper(), 'sl': f"{sl_price:.2f}"}, section="MINER")
                    break
                else:
                    log_with_format('warning', "Lệnh SL không mở được: {status}", 
                                   variables={'status': sl_status['status']}, section="MINER")
            except Exception as e:
                log_with_format('error', "Lỗi đặt SL (lần {attempt}/{max}): {error}",
                               variables={'attempt': str(attempt+1), 'max': str(max_retries), 'error': str(e)},
                               section="MINER")
                if attempt == max_retries - 1:
                    raise Exception("Không thể đặt SL sau 3 lần thử")
                await asyncio.sleep(wait_time)

        # Đặt Take Profit
        tp_order_id = None
        for attempt in range(max_retries):
            try:
                tp_order = await exchange.create_order(
                    symbol=SYMBOL,
                    type='TAKE_PROFIT_MARKET',
                    side='sell' if side == 'buy' else 'buy',
                    amount=quantity,
                    params={
                        'stopPrice': tp_price,
                        'reduceOnly': True,
                        'positionSide': 'BOTH'
                    }
                )
                tp_order_id = tp_order['id']
                # Kiểm tra trạng thái lệnh TP
                tp_status = await exchange.fetch_order(tp_order_id, SYMBOL)
                if tp_status['status'] == 'open':
                    position['tp_order_id'] = tp_order_id
                    log_with_format('info', "Đã đặt TP cho vị thế {side}: TP={tp}",
                                   variables={'side': side.upper(), 'tp': f"{tp_price:.2f}"}, section="MINER")
                    break
                else:
                    log_with_format('warning', "Lệnh TP không mở được: {status}", 
                                   variables={'status': tp_status['status']}, section="MINER")
            except Exception as e:
                log_with_format('error', "Lỗi đặt TP (lần {attempt}/{max}): {error}",
                               variables={'attempt': str(attempt+1), 'max': str(max_retries), 'error': str(e)},
                               section="MINER")
                if attempt == max_retries - 1:
                    raise Exception("Không thể đặt TP sau 3 lần thử")
                await asyncio.sleep(wait_time)

        # Xác nhận TP/SL đã được đặt thành công
        if not (sl_order_id and tp_order_id):
            log_with_format('error', "Không đặt được TP/SL, đóng vị thế ngay", section="MINER")
            close_side = 'sell' if side == 'buy' else 'buy'
            await close_position(close_side, quantity, entry_price, "TP/SL Failed")
            return None

        # Thông báo Telegram
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Mở vị thế {side.upper()}: Giá={entry_price:.2f}, "
                                                    f"SL={sl_price:.2f}, TP={tp_price:.2f}, Số lượng={quantity:.2f}")
        return order

    except Exception as e:
        log_with_format('error', "Lỗi nghiêm trọng khi đặt lệnh/TP/SL: {error}", 
                        variables={'error': str(e)}, section="MINER")
        if position:  # Đóng vị thế nếu đã mở nhưng TP/SL thất bại
            close_side = 'sell' if side == 'buy' else 'buy'
            for attempt in range(max_retries):
                try:
                    await close_position(close_side, quantity, entry_price, "Order Failed")
                    break
                except Exception as close_error:
                    log_with_format('error', "Lỗi đóng vị thế (lần {attempt}/{max}): {error}",
                                   variables={'attempt': str(attempt+1), 'max': str(max_retries), 'error': str(close_error)},
                                   section="MINER")
                    if attempt == max_retries - 1:
                        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] KHẨN CẤP: Không thể đóng vị thế {side.upper()} do lỗi: {str(close_error)}. Kiểm tra thủ công!")
                    await asyncio.sleep(wait_time)
        return None
        
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
   

import traceback

async def test_order_placement():
    global exchange, current_price, position
    log_with_format('info', "=== BẮT ĐẦU KIỂM TRA ĐẶT VỊ THẾ VÀ TP/SL ===", section="MINER")
    MIN_NOTIONAL_VALUE = 20.0  # Giá trị tối thiểu theo quy định Binance Futures
    wait_time = 5  # Thời gian chờ giữa các bước
    max_retries = 3
    monitoring_time = 120  # Thời gian theo dõi tối đa (2 phút)

    try:
        # Lấy thông tin symbol từ API Binance
        markets = await exchange.fetch_markets()
        try:
            symbol_info = next(m for m in markets if m['symbol'] == SYMBOL)
            precision = symbol_info['precision']['amount']
            # Đảm bảo precision là int
            if isinstance(precision, (int, float)):
                quantity_precision = int(precision)
            else:
                log_with_format('error', "Quantity precision không hợp lệ: {prec}", 
                                variables={'prec': str(precision)}, section="MINER")
                return False
            log_with_format('debug', "Thông tin symbol: Quantity Precision={prec}", 
                            variables={'prec': str(quantity_precision)}, section="MINER")
        except StopIteration:
            log_with_format('error', "Không tìm thấy thông tin symbol {symbol}", 
                            variables={'symbol': SYMBOL}, section="MINER")
            return False

        # Lấy giá hiện tại
        current_price = await get_price()
        if current_price is None:
            log_with_format('error', "Không thể lấy giá hiện tại để test", section="NET")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: Không lấy được giá hiện tại")
            return False

        # Tính TEST_QUANTITY đảm bảo notional >= 20
        TEST_QUANTITY = max(0.01, MIN_NOTIONAL_VALUE / current_price)
        log_with_format('debug', "Trước khi làm tròn: TEST_QUANTITY={qty}", 
                        variables={'qty': f"{TEST_QUANTITY:.6f}"}, section="MINER")
        # Đảm bảo quantity_precision là int trước khi dùng trong round()
        TEST_QUANTITY = round(float(TEST_QUANTITY), int(quantity_precision))
        notional_value = TEST_QUANTITY * current_price
        if notional_value < MIN_NOTIONAL_VALUE:
            log_with_format('debug', "Notional nhỏ hơn 20, điều chỉnh lại: {notional}", 
                            variables={'notional': f"{notional_value:.2f}"}, section="MINER")
            TEST_QUANTITY = round(float(MIN_NOTIONAL_VALUE / current_price), int(quantity_precision))
            notional_value = TEST_QUANTITY * current_price

        log_with_format('info', "Thông số test: Giá={price}, Số lượng={qty}, Giá trị={notional}",
                        variables={'price': f"{current_price:.2f}", 'qty': f"{TEST_QUANTITY:.{quantity_precision}f}", 
                                   'notional': f"{notional_value:.2f}"}, section="MINER")

        if notional_value < MIN_NOTIONAL_VALUE:
            log_with_format('error', "Không thể tạo TEST_QUANTITY hợp lệ: Notional={notional} < 20", 
                            variables={'notional': f"{notional_value:.2f}"}, section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: Giá trị danh nghĩa {notional_value:.2f} < 20")
            return False

        # Mở vị thế test với lệnh BUY
        log_with_format('info', "Đặt vị thế BUY để test", section="MINER")
        test_order = await place_order_with_tp_sl(
            side='buy',
            price=current_price,
            quantity=TEST_QUANTITY,
            volatility=0.0,
            predicted_price=current_price * 1.02,
            atr=0
        )

        if test_order is None or not position:
            log_with_format('error', "Không thể mở vị thế test BUY", section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: Không mở được vị thế BUY")
            return False

        # Đợi để sàn xử lý
        await asyncio.sleep(wait_time)

        # Kiểm tra trạng thái vị thế và TP/SL
        positions = await exchange.fetch_positions([SYMBOL])
        current_position = next((p for p in positions if p['symbol'] == SYMBOL), None)
        if not current_position or float(current_position['info']['positionAmt']) == 0:
            log_with_format('error', "Vị thế test không tồn tại trên sàn", section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: Vị thế không tồn tại")
            return False

        # Kiểm tra TP/SL trên sàn
        sl_exists = False
        tp_exists = False
        if 'sl_order_id' in position and position['sl_order_id']:
            try:
                sl_order = await exchange.fetch_order(position['sl_order_id'], SYMBOL)
                sl_exists = sl_order['status'] == 'open'
                log_with_format('info', "Trạng thái SL: {status}, Giá SL: {sl_price}",
                                variables={'status': sl_order['status'], 'sl_price': f"{position['sl_price']:.2f}"}, 
                                section="MINER")
            except Exception as e:
                log_with_format('error', "Lỗi kiểm tra SL: {error}", variables={'error': str(e)}, section="MINER")

        if 'tp_order_id' in position and position['tp_order_id']:
            try:
                tp_order = await exchange.fetch_order(position['tp_order_id'], SYMBOL)
                tp_exists = tp_order['status'] == 'open'
                log_with_format('info', "Trạng thái TP: {status}, Giá TP: {tp_price}",
                                variables={'status': tp_order['status'], 'tp_price': f"{position['tp_price']:.2f}"}, 
                                section="MINER")
            except Exception as e:
                log_with_format('error', "Lỗi kiểm tra TP: {error}", variables={'error': str(e)}, section="MINER")

        if not (sl_exists and tp_exists):
            log_with_format('error', "TP/SL không được đặt thành công: SL={sl_status}, TP={tp_status}",
                            variables={'sl_status': str(sl_exists), 'tp_status': str(tp_exists)}, section="MINER")
            await close_position('sell', TEST_QUANTITY, current_price, "Test Failed: TP/SL Missing")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: SL={sl_exists}, TP={tp_exists}")
            return False

        # Theo dõi giá để kiểm tra TP/SL
        log_with_format('info', "Bắt đầu theo dõi giá để kiểm tra TP/SL trong {time}s",
                        variables={'time': str(monitoring_time)}, section="MINER")
        start_time = time.time()
        tp_triggered = False
        sl_triggered = False

        while time.time() - start_time < monitoring_time:
            current_price = await get_price()
            if current_price is None:
                continue

            log_with_format('debug', "Giá hiện tại: {price} | SL={sl} | TP={tp}",
                            variables={'price': f"{current_price:.2f}", 'sl': f"{position['sl_price']:.2f}", 
                                       'tp': f"{position['tp_price']:.2f}"}, section="MINER")

            positions = await exchange.fetch_positions([SYMBOL])
            current_position = next((p for p in positions if p['symbol'] == SYMBOL), None)

            if not current_position or float(current_position['info']['positionAmt']) == 0:
                if 'tp_order_id' in position:
                    tp_order = await exchange.fetch_order(position['tp_order_id'], SYMBOL)
                    if tp_order['status'] in ['closed', 'filled']:
                        tp_triggered = True
                        log_with_format('info', "TP đã kích hoạt tại giá {price}", 
                                        variables={'price': f"{current_price:.2f}"}, section="MINER")
                        break
                if 'sl_order_id' in position:
                    sl_order = await exchange.fetch_order(position['sl_order_id'], SYMBOL)
                    if sl_order['status'] in ['closed', 'filled']:
                        sl_triggered = True
                        log_with_format('info', "SL đã kích hoạt tại giá {price}", 
                                        variables={'price': f"{current_price:.2f}"}, section="MINER")
                        break
                break

            await asyncio.sleep(5)

        # Đánh giá kết quả
        if tp_triggered:
            log_with_format('info', "Kiểm tra thành công: TP đã kích hoạt đúng", section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thành công: TP kích hoạt tại {current_price:.2f}")
            position = None
            return True
        elif sl_triggered:
            log_with_format('info', "Kiểm tra thành công: SL đã kích hoạt đúng", section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thành công: SL kích hoạt tại {current_price:.2f}")
            position = None
            return True
        else:
            log_with_format('warning', "TP/SL không kích hoạt trong thời gian theo dõi, đóng vị thế thủ công", section="MINER")
            for attempt in range(max_retries):
                try:
                    await close_position('sell', TEST_QUANTITY, current_price, "Test Completed: Manual Close")
                    log_with_format('info', "Đã đóng vị thế test thủ công", section="MINER")
                    break
                except Exception as e:
                    log_with_format('error', "Lỗi đóng vị thế test (lần {attempt}/{max}): {error}",
                                    variables={'attempt': str(attempt+1), 'max': str(max_retries), 'error': str(e)},
                                    section="MINER")
                    if attempt == max_retries - 1:
                        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: Không đóng được vị thế test")
                        return False
                    await asyncio.sleep(wait_time)

            log_with_format('info', "Kiểm tra hoàn tất: TP/SL được đặt thành công nhưng không kích hoạt trong {time}s",
                            variables={'time': str(monitoring_time)}, section="MINER")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test hoàn tất: TP/SL được đặt nhưng không kích hoạt trong {monitoring_time}s")
            return True

    except Exception as e:
        log_with_format('error', "Lỗi nghiêm trọng trong kiểm tra: {error}\nTrace: {trace}", 
                        variables={'error': str(e), 'trace': traceback.format_exc()}, section="MINER")
        if position:
            for attempt in range(max_retries):
                try:
                    await close_position('sell', TEST_QUANTITY, current_price, "Test Failed: Exception")
                    break
                except Exception as close_error:
                    log_with_format('error', "Lỗi đóng vị thế khi test thất bại: {error}", 
                                    variables={'error': str(close_error)}, section="MINER")
                    if attempt == max_retries - 1:
                        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] KHẨN CẤP: Test thất bại và không đóng được vị thế: {str(e)}")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Test thất bại: {str(e)}")
        return False
    finally:
        # Đóng kết nối exchange để tránh "Unclosed connector"
        try:
            await exchange.close()
            log_with_format('info', "Đã đóng kết nối với Binance", section="NET")
        except Exception as close_error:
            log_with_format('error', "Lỗi khi đóng kết nối: {error}", 
                            variables={'error': str(close_error)}, section="NET")
            

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
    global position
    max_retries = 3

    try:
        # Đặt lệnh thị trường
        for attempt in range(max_retries):
            try:
                order = await exchange.create_order(
                    symbol=SYMBOL,
                    type='MARKET',
                    side=side,
                    amount=quantity
                )
                entry_price = float(order['price']) if order.get('price') else price
                log_with_format('info', "Đã mở vị thế {side}: Giá={price}, Số lượng={qty}", 
                               variables={'side': side.upper(), 'price': f"{entry_price:.2f}", 'qty': f"{quantity:.2f}"}, section="MINER")
                break
            except Exception as e:
                log_with_format('error', "Lỗi đặt lệnh {side} (lần {attempt}/{max}): {error}", 
                               variables={'side': side, 'attempt': str(attempt+1), 'max': str(max_retries), 'error': str(e)}, section="MINER")
                if attempt == max_retries - 1:
                    raise Exception("Không thể mở vị thế sau 3 lần thử")
                await asyncio.sleep(2)

        # Tính TP và SL
        sl_price = entry_price * (1 + STOP_LOSS_PERCENT + volatility) if side == 'buy' else \
                   entry_price * (1 - STOP_LOSS_PERCENT - volatility)
        tp_price = entry_price * (1 - TAKE_PROFIT_PERCENT - volatility) if side == 'buy' else \
                   entry_price * (1 + TAKE_PROFIT_PERCENT + volatility)

        # Đặt SL trên Binance
        sl_success = False
        for attempt in range(max_retries):
            try:
                sl_order = await exchange.create_order(
                    symbol=SYMBOL,
                    type='STOP_MARKET',
                    side='sell' if side == 'buy' else 'buy',
                    amount=quantity,
                    params={'stopPrice': sl_price, 'reduceOnly': True}
                )
                sl_success = True
                break
            except Exception as e:
                log_with_format('error', "Lỗi đặt SL (lần {attempt}/{max}): {error}", 
                               variables={'attempt': str(attempt+1), 'max': str(max_retries), 'error': str(e)}, section="MINER")
                if attempt == max_retries - 1:
                    raise Exception("Không thể đặt SL sau 3 lần thử")
                await asyncio.sleep(2)

        # Đặt TP trên Binance
        tp_success = False
        for attempt in range(max_retries):
            try:
                tp_order = await exchange.create_order(
                    symbol=SYMBOL,
                    type='TAKE_PROFIT_MARKET',
                    side='sell' if side == 'buy' else 'buy',
                    amount=quantity,
                    params={'stopPrice': tp_price, 'reduceOnly': True}
                )
                tp_success = True
                break
            except Exception as e:
                log_with_format('error', "Lỗi đặt TP (lần {attempt}/{max}): {error}", 
                               variables={'attempt': str(attempt+1), 'max': str(max_retries), 'error': str(e)}, section="MINER")
                if attempt == max_retries - 1:
                    raise Exception("Không thể đặt TP sau 3 lần thử")
                await asyncio.sleep(2)

        # Kiểm tra chắc chắn TP/SL được đặt
        if not (sl_success and tp_success):
            log_with_format('error', "Không đặt được TP/SL trên Binance, đóng vị thế ngay", section="MINER")
            await close_position('sell' if side == 'buy' else 'buy', quantity, entry_price, "TP/SL Failed")
            return None

        # Lưu thông tin vị thế
        position = {
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'sl_order_id': sl_order['id'],
            'tp_order_id': tp_order['id']
        }
        log_with_format('info', "Đã đặt TP/SL trên Binance: SL={sl}, TP={tp}", 
                       variables={'sl': f"{sl_price:.2f}", 'tp': f"{tp_price:.2f}"}, section="MINER")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Mở vị thế {side.upper()}: Giá={entry_price:.2f}, "
                                                    f"SL={sl_price:.2f}, TP={tp_price:.2f}, Số lượng={quantity:.2f}")
        return order

    except Exception as e:
        log_with_format('error', "Lỗi nghiêm trọng khi đặt lệnh: {error}", variables={'error': str(e)}, section="MINER")
        if position:  # Đóng vị thế nếu đã mở nhưng TP/SL thất bại
            await close_position('sell' if side == 'buy' else 'buy', quantity, entry_price, "Order Failed")
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

    is_trading = False
    position = None
    data_buffer = []
    last_retrain_time = time.time()
    last_check_time = time.time()
    last_pnl_check_time = time.time()
    last_position_check = time.time()
    current_price = None
    last_price = None

    log_with_format('info', "Tải lịch sử hiệu suất từ cơ sở dữ liệu", section="CPU")
    load_historical_performance()

    if os.path.exists('lstm_model.keras'):
        lstm_model = load_model('lstm_model.keras')
        log_with_format('info', "Đã tải mô hình LSTM Regression từ file", section="CPU")
    else:
        lstm_model = create_lstm_model()
        log_with_format('info', "Đã tạo mới mô hình LSTM Regression", section="CPU")

    if os.path.exists('lstm_classification_model.keras'):
        lstm_classification_model = load_model('lstm_classification_model.keras')
        log_with_format('info', "Đã tải mô hình LSTM Classification từ file", section="CPU")
    else:
        lstm_classification_model = create_lstm_classification_model()
        log_with_format('info', "Đã tạo mới mô hình LSTM Classification", section="CPU")

    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        log_with_format('info', "Đã tải scaler từ file", section="CPU")
    else:
        scaler = MinMaxScaler()
        log_with_format('info', "Đã tạo mới scaler", section="CPU")

    if os.path.exists('rf_classifier.pkl'):
        with open('rf_classifier.pkl', 'rb') as f:
            rf_classifier = pickle.load(f)
        log_with_format('info', "Đã tải RandomForest Classifier từ file", section="CPU")
    else:
        rf_classifier = create_rf_classifier()
        log_with_format('info', "Đã tạo mới RandomForest Classifier", section="CPU")

    historical_data = await get_historical_data()
    if historical_data is None:
        log_with_format('error', "Không thể lấy dữ liệu lịch sử, thoát", section="NET")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Không thể lấy dữ liệu lịch sử, bot thoát")
        return
    closes, volumes, atr, (historical_closes, historical_volumes, historical_highs, historical_lows, ohlcv) = historical_data
    log_with_format('debug', "Kích thước dữ liệu ban đầu: closes={c_shape}, volumes={v_shape}",
                   variables={'c_shape': str(closes.shape), 'v_shape': str(volumes.shape)}, section="NET")

    init_db()
    if not os.path.exists('lstm_model.keras'):
        log_with_format('info', "Huấn luyện ban đầu mô hình AI", section="CPU")
        await train_advanced_model(ohlcv, historical_closes, historical_highs, historical_lows, initial=True)
    else:
        log_with_format('info', "Cập nhật mô hình AI", section="CPU")
        await train_advanced_model(ohlcv, historical_closes, historical_highs, historical_lows, initial=False)

    try:
        await exchange.set_leverage(LEVERAGE, SYMBOL)
        log_with_format('info', "Đã đặt đòn bẩy {leverage}x cho {symbol}", 
                        variables={'leverage': str(LEVERAGE), 'symbol': SYMBOL}, section="NET")
    except Exception as e:
        log_with_format('error', "Lỗi đặt đòn bẩy: {error}", variables={'error': str(e)}, section="NET")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Lỗi đặt đòn bẩy: {str(e)}")
        return

    asyncio.create_task(watch_position_and_price())

    # Thực hiện kiểm tra lệnh test
    log_with_format('info', "Bắt đầu kiểm tra lệnh test với số lượng nhỏ (0.01 ETH)", section="MINER")
    test_success = await test_order_placement()
    if not test_success:
        log_with_format('warning', "Kiểm tra lệnh test thất bại. Vui lòng kiểm tra thủ công và sửa lỗi trước khi tiếp tục!", section="MINER")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Kiểm tra lệnh test thất bại. Vui lòng kiểm tra thủ công!")
        return

    log_with_format('info', "Kiểm tra lệnh test thành công. Bạn có muốn tiếp tục chạy bot không? Gõ 'yes' trong Telegram để xác nhận.", section="MINER")
    await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Kiểm tra lệnh test thành công. Gõ 'yes' để chạy bot, hoặc bot sẽ dừng sau 60 giây.")

    # Chờ xác nhận từ Telegram
    confirmation = None
    start_time = time.time()
    while time.time() - start_time < 60:  # Chờ tối đa 60 giây
        updates = await bot.get_updates(offset=-1)
        for update in updates:
            if update.message and update.message.text.lower() == 'yes' and update.message.chat.id == int(CHAT_ID):
                confirmation = True
                break
        if confirmation:
            break
        await asyncio.sleep(1)

    if confirmation:
        log_with_format('info', "Xác nhận thành công. Bắt đầu chạy bot...", section="CPU")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Bot bắt đầu chạy...")
    else:
        log_with_format('warning', "Không có xác nhận sau 60 giây. Bot dừng lại.", section="CPU")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Bot dừng vì không có xác nhận.")
        return

    # Tiếp tục vòng lặp chính
    while True:
        current_time = time.time()
        if current_time - last_check_time < CHECK_INTERVAL:
            await asyncio.sleep(CHECK_INTERVAL - (current_time - last_check_time))
            continue
        last_check_time = current_time

        if current_price is None:
            log_with_format('warning', "Chưa có giá từ polling, chờ 5s", section="NET")
            await asyncio.sleep(5)
            continue
        
        if position:
            log_with_format('debug', "Đang có vị thế {side}, chờ đóng trước khi giao dịch tiếp",
                           variables={'side': position['side'].upper()}, section="MINER")
            continue
        
        if last_price is None:
            last_price = current_price

        try:
            balance_info = await exchange.fetch_balance(params={'type': 'future'})
            available_balance = float(balance_info['info']['availableBalance'])
            log_with_format('debug', "Số dư khả dụng: {balance} USDT", 
                            variables={'balance': f"{available_balance:.2f}"}, section="CPU")
        except Exception as e:
            log_with_format('warning', "Lỗi lấy số dư: {error}, dùng giá trị mặc định 0", 
                            variables={'error': str(e)}, section="CPU")
            available_balance = 0

        if ENABLE_PNL_CHECK and position and current_time - last_pnl_check_time >= PNL_CHECK_INTERVAL:
            unrealized_pnl = (current_price - position['entry_price']) * position['quantity'] * LEVERAGE / position['entry_price'] \
                            if position['side'].lower() == 'buy' else \
                            (position['entry_price'] - current_price) * position['quantity'] * LEVERAGE / position['entry_price']
            if unrealized_pnl < -PNL_THRESHOLD:
                log_with_format('warning', "PNL vượt ngưỡng: {pnl}, đóng vị thế", 
                                variables={'pnl': f"{unrealized_pnl:.2f}"}, section="MINER")
                await close_position('sell' if position['side'].lower() == 'buy' else 'buy', 
                                   position['quantity'], current_price, "PNL Threshold")
            last_pnl_check_time = current_time

        if performance['profit'] < -DAILY_LOSS_LIMIT or daily_trades >= MAX_DAILY_TRADES or performance['consecutive_losses'] >= 3:
            log_with_format('warning', "DỪNG BOT: Profit={profit} | Trades={trades} | Losses liên tiếp={losses}",
                           variables={'trades': str(daily_trades), 'losses': str(performance['consecutive_losses'])}, 
                           profit=performance['profit'], section="CPU")
            await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Bot dừng: Profit={performance['profit']:.2f}, "
                                                        f"Trades={daily_trades}, Losses liên tiếp={performance['consecutive_losses']}")
            break

        historical_data = await get_historical_data()
        if historical_data is None:
            log_with_format('warning', "Không lấy được dữ liệu, chờ 10s", section="NET")
            await asyncio.sleep(10)
            continue
        closes, volumes, atr, (historical_closes, historical_volumes, historical_highs, historical_lows, ohlcv) = historical_data
        log_with_format('debug', "Kích thước dữ liệu mới: closes={c_shape}, volumes={v_shape}",
                       variables={'c_shape': str(closes.shape), 'v_shape': str(volumes.shape)}, section="NET")

        data_buffer.extend(ohlcv)
        if len(data_buffer) > BUFFER_SIZE:
            data_buffer = data_buffer[-BUFFER_SIZE:]

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
        vwap = calculate_vwap(ohlcv)
        volume_spike = volumes[-1] > (np.mean(volumes[-10:-1]) * VOLUME_SPIKE_THRESHOLD) if len(volumes) > 10 else False

        ohlcv_120s, _ = await get_historical_data_multi_timeframe('2m', 5)
        candle_pattern = detect_candle_patterns(ohlcv_120s) if ohlcv_120s else None

        stoch_k, stoch_d = calculate_stochastic_rsi(historical_closes) or (50, 50)
        log_with_format('debug', "Stochastic RSI: K={k}, D={d}",
                       variables={'k': f"{stoch_k:.2f}", 'd': f"{stoch_d:.2f}"}, section="CHỈ BÁO")

        prediction_result = await predict_price_and_confidence(
            closes, volumes, atr, historical_closes, historical_highs, historical_lows, historical_volumes, buy_score=0, sell_score=0
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

        buy_score = 0
        sell_score = 0
        active_strategies = []
        for strategy in STRATEGIES:
            wins = strategy_performance[strategy.name]['wins']
            losses = strategy_performance[strategy.name]['losses']
            total = wins + losses
            if total >= 50 and wins / total < 0.4:
                log_with_format('warning', "Tạm dừng chiến lược {name}: Win Rate={win_rate}",
                                variables={'name': strategy.name, 'win_rate': f"{wins/total:.2%}"}, section="CHIẾN LƯỢC")
                await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Tạm dừng chiến lược {strategy.name}: "
                                                            f"Win Rate={wins/total:.2%} sau {total} giao dịch")
                continue
            dynamic_weight = strategy.weight * (wins / total if total > 0 else 1.0)
            log_with_format('debug', "Chiến lược {name}: Win Rate={win_rate}, Weight={weight}",
                           variables={'name': strategy.name, 'win_rate': f"{wins/total:.2%}" if total > 0 else "N/A", 
                                      'weight': f"{dynamic_weight:.2f}"}, section="CHIẾN LƯỢC")

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
                'volatility': volatility,
                'vwap': vwap,
                'candle_pattern': candle_pattern
            }
            if strategy.evaluate_buy(**kwargs):
                buy_score += dynamic_weight
                active_strategies.append(strategy.name)
            if strategy.evaluate_sell(**kwargs):
                sell_score += dynamic_weight
                active_strategies.append(strategy.name)

        buy_score += (confidence_buy * 50)
        sell_score += (confidence_sell * 50)
        log_with_format('info', "Điểm mua: {buy}, Điểm bán: {sell}",
                       variables={'buy': f"{buy_score:.2f}", 'sell': f"{sell_score:.2f}"}, section="CHIẾN LƯỢC")

        historical_closes_5m, _ = await get_historical_data_multi_timeframe('5m', 20)
        historical_closes_15m, _ = await get_historical_data_multi_timeframe('15m', 20)
        if historical_closes_5m is not None and historical_closes_15m is not None:
            ema_short_5m = np.mean(historical_closes_5m[-5:])
            ema_long_5m = np.mean(historical_closes_5m[-15:])
            ema_short_15m = np.mean(historical_closes_15m[-5:])
            ema_long_15m = np.mean(historical_closes_15m[-15:])

        quantity = min(BASE_AMOUNT, (available_balance * USE_PERCENTAGE * LEVERAGE) / current_price)
        if confidence_buy >= MIN_CONFIDENCE and confidence_buy > confidence_sell and buy_score >= BUY_THRESHOLD:
            await place_order_with_tp_sl('buy', current_price, quantity, volatility, predicted_price, atr)
            daily_trades += 1
        elif confidence_sell >= MIN_CONFIDENCE and confidence_sell > confidence_buy and sell_score >= SELL_THRESHOLD:
            await place_order_with_tp_sl('sell', current_price, quantity, volatility, predicted_price, atr)
            daily_trades += 1

        last_price = current_price
        await asyncio.sleep(CHECK_INTERVAL)

async def shutdown_bot(reason, error=None):
    try:
        log_with_format('info' if not error else 'error', f"Bot dừng: {reason}" + (f" - Lỗi: {error}" if error else ""), 
                        variables={'error': str(error)} if error else None, section="NET")
        await bot.send_message(chat_id=CHAT_ID, text=f"[{SYMBOL}] Bot dừng: {reason}" + (f" - Lỗi: {error}" if error else ""))
    except Exception as telegram_error:
        print(f"Lỗi gửi thông báo Telegram khi dừng bot: {telegram_error}")

def get_user_choice():
    print("\n=== CHỌN CHẾ ĐỘ CHẠY BOT ===")
    print("1: Test đặt vị thế (kiểm tra TP/SL)")
    print("2: Chạy giao dịch tự động (Auto Trade)")
    print("Nhấn phím bất kỳ khác để thoát")
    
    while True:
        choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()
        if choice in ['1', '2']:
            return choice
        else:
            print("Lựa chọn không hợp lệ. Thoát chương trình.")
            return None

async def run_bot():
    global exchange
    choice = get_user_choice()
    if choice is None:
        log_with_format('info', "Người dùng đã thoát chương trình", section="CPU")
        return
    
    try:
        if choice == '1':
            log_with_format('info', "Chạy chế độ test đặt vị thế", section="MINER")
            await test_order_placement()
        elif choice == '2':
            log_with_format('info', "Chạy chế độ giao dịch tự động", section="CPU")
            await optimized_trading_bot()
    finally:
        # Đóng kết nối exchange khi bot hoàn tất
        try:
            await exchange.close()
            log_with_format('info', "Đã đóng kết nối với Binance", section="NET")
        except Exception as e:
            log_with_format('error', "Lỗi khi đóng kết nối: {error}", 
                            variables={'error': str(e)}, section="NET")

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        asyncio.run(shutdown_bot("Người dùng dừng bot"))
    except Exception as e:
        asyncio.run(shutdown_bot("Lỗi bot", e))
