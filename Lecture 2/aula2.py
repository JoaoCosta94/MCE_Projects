# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:40:19 2015

@author: beatrizsilveira
"""

from scipy.integrate import odeint
from pylab import*

gama=0
o=100

#odeint method    
def deriv(x,t):
    return array([x[1],-x[0]-gama*x[1]])
    
time=linspace(0.0,10.0,100)
xinit=array([0.0005,0.2])
x_odeint=odeint(deriv,xinit,time)

#euler method
h=time[1]-time[0]
x_e=zeros((o,2),dtype=float)
x_e[0,:]=xinit

for i in range(len(time)-1):
    x_e[i+1,0]=x_e[i,0]+h*x_e[i,1]
    x_e[i+1,1]=x_e[i,1]+h*(-x_e[i,0]-gama*x_e[i,1])
    
#euler cromer method
x_ec=zeros((o,2),dtype=float)   #array com 1000 elementos e cade um tem 2 parâmtros (posição e velocidade, neste caso)
x_ec[0,:]=xinit

for i in range(len(time)-1):
    x_ec[i+1,0]=x_ec[i,0]+h*x_ec[i,1]
    x_ec[i+1,1]=x_ec[i,1]+h*(-x_ec[i+1,0]-gama*x_ec[i,1])
    

print(x_ec[-1,0])
figure()
#plot(time,x_odeint[:,0])
#plot(time,x_ec[:,0])
#plot(time,x_e[:,0])
plot(time,x_odeint[:,0]**2+x_odeint[:,1]**2)
plot(time,x_e[:,0]**2+x_e[:,1]**2)
plot(time,x_ec[:,0]**2+x_ec[:,1]**2)
xlabel('t')
ylabel('x')
show()