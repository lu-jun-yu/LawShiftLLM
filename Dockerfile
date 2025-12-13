FROM continuumio/miniconda3

COPY environment_ori.yml /tmp/environment_ori.yml
COPY requirements.txt /tmp/requirements.txt

# 创建环境
RUN conda env create -f /tmp/environment_ori.yml && conda clean -a -y

# 【修正点】使用 /opt/conda 路径，且确保环境名与 yml 文件一致
RUN /opt/conda/envs/llm_ori/bin/pip install -r /tmp/requirements.txt --default-timeout=1000

# 【修正点】确保 PATH 里的环境名也是 llm_ori
ENV PATH /opt/conda/envs/llm_ori/bin:$PATH

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