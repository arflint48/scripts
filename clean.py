import os
import sys
import re
import math
import array
import subprocess as sp
#import argparse

#def enter_node(node):
    #sp.Popen(["ssh -T", node, "&&", "rm -r", temp, "&&", "exit"], shell=True, stdout=sp.PIPE, stderr=sp.PIPE).communicate()
    #print("entering" + node)

user = sp.run(["whoami"],capture_output=True)
user = str(user.stdout)
user="".join(ch for ch in user if ch.isalnum())
user = user.strip("bn")
print(user)

temp = "/tmp/" + user + "/"
temp.replace(" ","")

active = sp.run(['qstat', '-u', user],capture_output=True)
active = (active.stdout)
active = active.split()

nodelist = []
len_q = len(active)
for i in range(0,len_q):
    line = str(active[i])
    p1 = re.compile('\d{1,8}[.]maple')
    res = p1.findall(line)
    if len(res)!=0:
        res = "".join(str(element) for element in res)
        node = sp.run(['qstat', '-f', res],capture_output=True)
        node = str((node.stdout))
        p2 = re.compile('cn\d{3}')
        cn = p2.findall(node)
        if len(cn)!=0:
            cn = cn[0]
            nodelist.append(cn)

num_nodes = len(nodelist)
usable_nodes = []

i = 30
while i<=140:
    for x in range(0,num_nodes):
        fn = nodelist[x]
        fn = "".join(str(element) for element in fn)
        fn = fn.strip('cn')
        fn = int(fn)
        if i == fn:
            break
        elif x == (num_nodes - 1) and i != fn:
            usable_nodes.append(i)
    i+=1

#print(usable_nodes)
num_nodes = len(usable_nodes)

j=0
while j < num_nodes:
    if usable_nodes[j] <= 99:
        node = "cn" + "0" + str(usable_nodes[j])
    else:
        node = "cn" + str(usable_nodes[j])
    #enter_node(node)
    with sp.Popen(['ssh', '-T', node], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True) as p:
        output, error = p.communicate("""
        cd /tmp/
        rm -r %s/
        exit
        """ %user)
        print(output)
        print(error)
        print(p.returncode)
    j += 1

