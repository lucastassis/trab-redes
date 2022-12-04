#!/usr/bin/env python3
# encoding: utf-8
"""Use instead of `python3 -m http.server` when you need CORS"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
from threading import Timer
import sys
import os
import psutil


class repeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args,**self.kwargs)

def job(sname):
    f = open(f'monitor-{sname}.txt', 'a+')
    now = datetime.now()
    now = now.strftime("%H:%M:%S")
    usage = psutil.cpu_percent(0.0)
    memory = psutil.virtual_memory()
    net = psutil.net_io_counters()
    # temperature = CPUTemperature()
    p = psutil.Process()
    io_counters = p.io_counters() 
    disk_usage_process = io_counters[2] + io_counters[3]
    disk_io_counter = psutil.disk_io_counters()
    disk_total = disk_io_counter[2] + disk_io_counter[3]
    disk_percentage = (disk_usage_process/disk_total) * 100
    f.write(str(now) + "," +
            str(usage) + "," + 
            str(memory[3]) + "," +
            str(net[0]) + "," +
            str(net[1]) + "," +
            str(disk_percentage) + "\n")
    f.close()


def get_cpu_data(sname) -> None:
    f = open(f'monitor-{sname}.txt', 'w')
    f.write("Time,CPU_use,RAM_filled,Net_bytes_sent,Net_bytes_received,Disk_IO_percentage\n")
    f.close()
    monitor = repeatTimer(1, job, (sname,))
    monitor.start()

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(CORSRequestHandler, self).end_headers()

def main(args):
    get_cpu_data(args[2])
    httpd = HTTPServer((args[1], 8000), CORSRequestHandler)
    httpd.serve_forever()
if __name__ == '__main__':
    main(sys.argv)
    