FROM tensorflow/tensorflow:1.12.0-py3

WORKDIR /project

# Update and install vim and zsh
RUN apt-get update \
 && apt-get install -y wget \
 && apt-get install -y git \
 && apt-get install -q -y vim zsh libsm6 libxext6 libxrender-dev

# Install python requirements
ADD requirements.txt ./
RUN pip install -r ./requirements.txt && \
    rm ./requirements.txt

# Change Default Shell to ZSH
RUN chsh -s $(which zsh)
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

# Install syntax highlight and auto suggestions
RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Add plugins to ~/.zshrc
RUN sed -i "s/  git/  git history zsh-autosuggestions zsh-syntax-highlighting extract cp colored-man-pages colorize command-not-found/"  ~/.zshrc
RUN sed -i "s/ZSH_THEME=\"robbyrussell\"/ZSH_THEME=\"af-magic\"/"  ~/.zshrc
SHELL ["/bin/bash", "-c"]

# Add locale support for chinese
RUN apt-get update --fix-missing \
 && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

CMD ["python", "app.py"]
