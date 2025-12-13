# 1. 使用 Miniconda 基础镜像，体积小
FROM continuumio/miniconda3

# 2. 将环境配置文件复制到镜像中
COPY environment_ori.yml /tmp/environment_ori.yml
COPY requirements.txt /tmp/requirements.txt

# 3. 创建环境并清理缓存（优化层，安装依赖是耗时操作）
# 将所有安装和清理操作放在一个 RUN 命令中，以减少镜像层数
RUN conda env create -f /tmp/environment_ori.yml && conda clean -a -y
# RUN conda init && conda activate llm

# 4. 激活新环境并在其中运行 Pip 安装 (更容易追踪 Pip 错误)
RUN /root/anaconda3/envs/llm_ori/bin/pip install -r /tmp/requirements.txt
# --no-cache-dir

# 5. 设置环境变量，确保容器启动时默认使用 Conda 环境
ENV PATH /opt/conda/envs/llm/bin:$PATH

# 6. 复制您的应用代码
COPY ./config /app/config
COPY ./eval /app/eval
COPY ./LawShift /app/LawShift
COPY ./results/*/summary.md /app/results/*/summary.md
COPY ./train /app/train
COPY ./__init__.py /app/__init__.py
COPY ./.gitignore /app/.gitignore
COPY ./check_weights.py /app/check_weights.py
COPY ./environment.yml /app/environment.yml
COPY ./label.json /app/label.json
COPY ./prompt_template.py /app/prompt_template.py
COPY ./requirements.txt /app/requirements.txt
COPY ./setup_resources.py /app/setup_resources.py
WORKDIR /app