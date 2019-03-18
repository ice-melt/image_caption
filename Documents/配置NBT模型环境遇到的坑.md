# 遇到的坑

1. 服务器环境配置的时候，安装程序显示memory错误

   解决方式：在安装的时候使用--no-cache-dir的命令，不保存安装缓存。

2. 服务器运行程序的时候报内存溢出错误

   解决方式，设置虚拟内存：

   ```bash
   ubuntu18.04默认的swap文件在根目录/下，名字是swapfile
   #1.查看交换分区大小
   free -m 
   #在创建完毕后也可以用这个命令查看内存情况
   #2.创建一个swap文件
   sudo dd if=/dev/zero of=swap bs=1024 count=4000000
   #创建的交换文件名是swap，后面的40000000是4g的意思，可以按照自己的需要更改
   #3.创建swap文件系统
   sudo mkswap -f swap
   #4.开启swap
   sudo swapon swap
   #5.关闭和删除原来的swapfile
   sudo swapoff  swapfile
   sudo rm /swapfile
   #6.设置开机启动
   sudo vim /etc/fstab
   ```


3.服务器在关闭终端时自动停止运行

解决方式：使用如下命令忽略挂起命令：

```bash
nohup python3 display.py &
```



 4.docker-ce安装时版本低，使得nvidia-docker安装报错

解决方式：直接使用官方维护的安装脚本进行安装

```bash
1.首先安装curl：
sudo apt-get install -y curl
2.然后使用docker自行维护的脚本来安装docker：
$ curl -fsSL https://get.docker.com -o get-docker.sh
$ sudo sh get-docker.sh
```



5.docker使用时需要连接vpn才能下载第七步的文件，需要配置proxy

使用如下命令：

```python
代理服务器可以在启动并运行后阻止与Web应用程序的连接。如果您位于代理服务器后面，请使用以下ENV命令将以下行添加到Dockerfile中，以指定代理服务器的主机和端口：
# Set proxy server, replace host:port with values for your servers
ENV http_proxy host:port
ENV https_proxy host:port
```

