# Código desenvolvido para disciplina de Laboratório de Redes: Predição de QoS de um Serviço de Streaming utilizando Aprendizado Federado

Essa documentação tem como objetivo apresentar brevemente a organização dos diretórios do código. Também tem um mini-guia de como utilizar o código. Informações mais teóricas do trabalho podem ser encontradas no relatório disponibilizado no repositório.

## Organização dos diretórios
`/src/player-x`: contém o código de cada um dos três players utilizados nos clientes. Esse código foi desenvolvido por https://github.com/leandrocalmeida/player-mpegdash.

`/src/results/final-results/`:  contém os diretórios com cada experimento gerado. Os arquivos `processed_X.csv` contém os dados processados referentes ao host X. O arquivo `process.py` é o script desenvolvido para concatenar os dados dos hosts (`player-staX.csv`) e switches(`monitor-swX.csv`).

`/src/scripts-federated-learning`: contém os scripts dos modelos de aprendizado federado.

`/src/scripts-regular-learning`: contém os scripts dos modelos de aprendizado de máquina MLP e RandomForest.

`/src/server.py`: é o script que coleta os dados dos switches.

`/src/topology.py`: é o script do Mininet-Wifi para utilizar a topologia da rede.

## Como rodar a rede

Primeiramente deve-se preparar o servidor Apache2 na VM do Mininet-Wifi, e preparar os arquivos que serão utilizados para streaming. Para isso sugerimos utilizar o filme Big Buck Bunny (https://peach.blender.org/).

Com o arquivo do filme, deve-se preparar os arquivos para streaming. Com isso, supondo que temos o arquivo `bbb.mp4`, utilizamos o comando:

``` 
ffmpeg -i bbb.mp4 -c:a aac -ac 2 -b:a 128k -vn bbb-audio-128k.mp4 && \
ffmpeg -i bbb.mp4 -c:a aac -ac 2 -b:a 64k -vn bbb-audio-64k.mp4 && \
ffmpeg -i bbb.mp4 -c:a aac -ac 2 -b:a 32k -vn bbb-audio-32k.mp4 && \
ffmpeg -i bbb.mp4 -an -r 18 -c:v libx264 -x264opts 'keyint=18:min-keyint=18:no-scenecut' -b:v 700k -maxrate 700k -bufsize 350k -vf 'scale=426:240' bbb_426x240_18_700k.mp4 && \
ffmpeg -i bbb.mp4 -an -r 24 -c:v libx264 -x264opts 'keyint=24:min-keyint=24:no-scenecut' -b:v 2100k -maxrate 2100k -bufsize 1050k -vf 'scale=854:480' bbb_854x480_24_2100k.mp4 && \
ffmpeg -i bbb.mp4 -an -r 30 -c:v libx264 -x264opts 'keyint=30:min-keyint=30:no-scenecut' -b:v 3760k -maxrate 3760k -bufsize 1880k -vf 'scale=1280:720' bbb_1280x720_30_3760k.mp4
```

Depois disso, utilizamos esse outro comando para gerar o arquivo de descrição:

```
MP4Box -dash 4000 -out bbb-mp4.mpd bbb_1280x720_30_3760k.mp4 bbb_854x480_24_2100k.mp4 bbb_426x240_18_700k.mp4 bbb-audio-32k.mp4 bbb-audio-64k.mp4 bbb-audio-128k.mp4
```

Uma vez que todos esses arquivos foram criados, basta mover para o diretório do server (exemplo: `/var/www/html`).

Com o servidor configurado, basta rodar o script da topologia com o comando `sudo python topology.py`. Uma vez dentro do mininet-wifi, basta abrir terminais para todos os switches e hosts com o comando `xterm h1 h2 h3 sw1 sw2 sw3`. Dentro do terminal de cada switch devemos iniciar a captura dos dados de rede e CPU utilizando o comando `python server.py swX`, substituindo o "X" pelo número do switch. Para iniciar o stream no cliente, basta entrar nos terminais dos hosts e utilizar o comando `nodejs ./player-X/index.js`, substituindo o "X" pelo número do host. Ao fim da simulação, os arquivos de dados do servidor e hosts serão gerados e os scripts de processamento poderão ser utilizados. 


