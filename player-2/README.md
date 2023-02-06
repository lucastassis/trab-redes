# Purpose of this repository
In this repository, you can find a [dash.js](https://github.com/Dash-Industry-Forum/dash.js/) player (MPEG-DASH compatible) with modifications to log video statistics. 

## Install the required software
To run the dash.js player, you will need to install a nodejs packet.
```bash
$ sudo apt-get install nodejs
```

## Getting started
If you already have the required software, you should clone this repository in your machine:
```bash
$ git clone https://github.com/leandrocalmeida/player-mpegdash.git
```

The next step is install [puppeter](https://www.npmjs.com/package/puppeteer), a nodejs library which provides a high-level API to control chrome browser.
```bash
$ cd player-mpegdash/
$ npm install
```

## Adjust parameters
1. In this repository, the default video source is available in: 
```html
url = https://dash.akamaized.net/akamai/bbb_30fps/bbb_30fps.mpd
```
If you need playing other video, you need to update line 67 of the index.html file.

2. You need to check the executable path of google chrome in your system.
```bash
$ which google-chrome
```
If the output of the command was different of "/usr/bin/google-chrome", you need to update line 14 of the index.js file.

3. Define the video duration (ms) in line 33 (videoDuration) of the index.js file. For example, if you need to play for 1 minute, you need to change videoDuration value to "60 * 1000"

## Run the dash.js player

```bash
$ nodejs index.js
```

## Log and results
You can see a new logfile for each execution of a dash.js in the current directory. This file has the following format: 

```
timestamp;droppedFrames;bufferLevel;frameRate;bitrate;calculatedBitrate;resolution
```
