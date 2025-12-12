# 1. 使用 Miniconda 基础镜像，体积小
FROM continuumio/miniconda3

# 2. 将环境配置文件复制到镜像中
COPY environment.yml /tmp/environment.yml

# 3. 创建环境并清理缓存（优化层，安装依赖是耗时操作）
# 将所有安装和清理操作放在一个 RUN 命令中，以减少镜像层数
RUN conda env create -f /tmp/environment.yml -n llm && \
    conda clean -a -y

# 4. 设置环境变量，确保容器启动时默认使用 Conda 环境
ENV PATH /opt/conda/envs/llm/bin:$PATH

# 5. 复制您的应用代码
COPY . /app
WORKDIR /app

# 6. 定义容器启动时的默认命令
CMD ["python", "main.py"]