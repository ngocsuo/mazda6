#!/bin/bash

# Kiểm tra quyền root
if [ "$(id -u)" -ne 0 ]; then
    echo "Vui lòng chạy script với quyền root hoặc sudo."
    exit 1
fi

# Xác định hệ điều hành
OS=""
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
elif [ -f /etc/redhat-release ]; then
    OS="centos"
else
    echo "Không thể xác định hệ điều hành. Hỗ trợ Ubuntu, Debian, và CentOS."
    exit 1
fi

install_docker_ubuntu_debian() {
    apt update
    apt install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/$(lsb_release -is | tr '[:upper:]' '[:lower:]')/gpg | apt-key add -
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/$(lsb_release -is | tr '[:upper:]' '[:lower:]') $(lsb_release -cs) stable"
    apt update
    apt install -y docker-ce docker-ce-cli containerd.io
    systemctl enable --now docker
}

install_docker_centos() {
    yum install -y yum-utils
    yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    yum install -y docker-ce docker-ce-cli containerd.io
    systemctl enable --now docker
}

case "$OS" in
    ubuntu|debian)
        install_docker_ubuntu_debian
        ;;
    centos)
        install_docker_centos
        ;;
    *)
        echo "Hệ điều hành không được hỗ trợ."
        exit 1
        ;;
esac

# Kiểm tra cài đặt Docker
if ! command -v docker &> /dev/null; then
    echo "Cài đặt Docker thất bại!"
    exit 1
fi

echo "Docker đã được cài đặt thành công!"
docker --version
