# 环境安装 
- 安装 Miniconda  
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe
- 运行 Anaconda Prompt (Miniconda3)

- 设置conda代理  
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/  
	conda config --set show_channel_urls yes

- 设置pip代理  
	pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

- 安装tensorflow  
	conda install tensorflow-gpu

- 更新  
	conda update --all  

- [测试环境](1_env.py)