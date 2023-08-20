import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
import math
from scipy import interpolate
from scipy.integrate import solve_ivp


data=np.loadtxt('300V.csv',delimiter=',',skiprows=9)

def mini(x_data,y_data):
    min_index=np.argmin(y_data)
    return x_data[min_index]

a=[]
b=[]
c=[]
def pfit(x_data,y_data):
    m=6.64215627*10**(-26)
    degree=2
    fit_x_data=[]
    fit_y_data=[]
    for j in x_data:
        if mini(x_data,y_data)-0.0001<=j and j<=mini(x_data,y_data)+0.0001:
            fit_x_data.append(j)
            fit_y_data.append(y_data[x_data.tolist().index(j)])#index
    coeffs=np.polyfit(fit_x_data,fit_y_data,degree)
    poly=np.poly1d(coeffs)
    #a.append(coeffs[0])
    #b.append(coeffs[1])
    #c.append(coeffs[2])
    #derivative=np.polyder(poly)
    #y_fit=poly(x_data)
    #mse=mean_squared_error(y_data,y_fit)
    #r2=r2_score(y_data,y_fit)
    return math.sqrt((2*coeffs[0])/m)

x_data=10**-3*data[:,2]
fre=[]
pos=[]
for i in range(31):
    y_data=1.6*10**-19*data[:,i+3]
    fre.append(pfit(x_data,y_data))
    pos.append(mini(x_data,y_data))


delta=[i for i in range(31)]
delta=np.array(delta
delta=10*delta
plt.plot(delta,10**6*np.array(fre))
plt.xlabel('voltage')
plt.ylabel('trap frequency')
plt.show()
'''
for i in range(13):#plot fitted potential and original potential
    x0=pos[i]
    a=-1.85*10**-3*x0**3-3.41*10**-7*x0**2-1.39*10**-11*x0+7.93*10**-14
    b=-3.44*10**-11*x0**2-1.61*10**-13*x0-4.72*10**-20
    c=4.91*10**-14*x0**2-3.46*10**-18*x0+1.6*10**-20
    plt.plot(10**6*x_data,a*x_data**2+b*x_data+c,color='b')
    plt.plot(x_data,1.6*10**-19*data[:,3+i],color='r')
plt.plot([],[],color='b',label='fit')
plt.plot([],[],color='r',label='original')
plt.xlabel('position/m')
plt.ylabel('potential')
plt.legend()
plt.show()


def ode_system(t,y,L,T):
    x,v=y
    x0=-L/T*t
    if 0<=t<=T:
        a=-1.85*10**-3*x0**3-3.41*10**-7*x0**2-1.39*10**-11*x0+7.93*10**-14
        b=-3.44*10**-11*x0**2-1.61*10**-13*x0-4.72*10**-20
    else:
        a=-1.85*10**-3*(-8*10**-5)**3-3.41*10**-7*(-8*10**-5)**2-1.39*10**-11*(-8*10**-5)+7.93*10**-14
        b=-3.44*10**-11*(-8*10**-5)**2-1.61*10**-13*(-8*10**-5)-4.72*10**-20
    dx_dt=v
    dv_dt=1/(6.64215627*10**(-26))*(-2*a*x-b)
    return [dx_dt,dv_dt]


for i in range(3):#plot x-t
    solution=solve_ivp(ode_system,(0,2*95*10**-5),[0,0],t_eval=np.linspace(0,2*95*10**-5,10000),args=(i+1,))
    x_values=solution.y[0]
    plt.plot(10**6*solution.t,10**6*x_values)
    #plt.plot(10**6*solution.t,10**6*(-(8/95)*10**(i+1)*solution.t),label=f'v={i+1}times')


def ode_system_2(t,y,L,T):
    x,v=y
    x0=-L/2*(1-math.cos(math.pi*t/T))
    if 0<=t<=T:
        a=-1.85*10**-3*x0**3-3.41*10**-7*x0**2-1.39*10**-11*x0+7.93*10**-14
        b=-3.44*10**-11*x0**2-1.61*10**-13*x0-4.72*10**-20
    else:
        a=-1.85*10**-3*(-8*10**-5)**3-3.41*10**-7*(-8*10**-5)**2-1.39*10**-11*(-8*10**-5)+7.93*10**-14
        b=-3.44*10**-11*(-8*10**-5)**2-1.61*10**-13*(-8*10**-5)-4.72*10**-20
    dx_dt=v
    dv_dt=1/(6.64215627*10**(-26))*(-2*a*x-b)
    return [dx_dt,dv_dt]

solution=solve_ivp(ode_system,(0,4*10**-4),[0,0],t_eval=np.linspace(0,4*10**-4,10000),args=(8*10**-5,2*10**-4))
x_values=solution.y[0]
plt.plot(10**6*solution.t,10**6*x_values,label='the position of the ion')
x0=[]
for i in solution.t:
    if i<95*10**-6:
        x0.append(-10**6*4*10**-5*(1-math.cos(math.pi*i/(95*10**-8))))
     
#plt.plot(10**6*solution.t[0:len(x0)],x0,color='b',label='x0')
#plt.plot(10**6*solution.t[len(x0):],[-80 for i in range(len(solution.t[len(x0):]))],color='b')



maximum=np.max(x_values[5000:])
minimum=np.min(x_values[5000:])
print(maximum-minimum)

  
for i in range(2):#plot E-x
    solution=solve_ivp(ode_system,(0,2*95*10**-6),[0,0],t_eval=np.linspace(0,2*95*10**-6,1000),args=(i+1,))
    E=[]
    T=[]
    U=[]
    t=solution.t
    x=solution.y[0]
    v=solution.y[1]
    for j in t:
        if 0<=j<=95*10**-6:
            x0=-8/95*10**(i+1)*j
            a=-1.85*10**-3*x0**3-3.41*10**-7*x0**2-1.39*10**-11*x0+7.93*10**-14
            b=-3.44*10**-11*x0**2-1.61*10**-13*x0-4.72*10**-20
            c=4.91*10**-14*x0**2-3.46*10**-18*x0+1.6*10**-20
            index=np.where(t==j)[0][0]
            pos=x[index]
            speed=v[index]
            E.append(a*pos**2+b*pos+c+0.5*6.64215627*10**(-26)*speed**2)
            T.append(0.5*6.64215627*10**(-26)*speed**2)
            U.append(a*pos**2+b*pos+c)
        else:
            a=-1.85*10**-3*(-8*10**-4)**3-3.41*10**-7*(-8*10**-4)**2-1.39*10**-11*(-8*10**-4)+7.93*10**-14
            b=-3.44*10**-11*(-8*10**-4)**2-1.61*10**-13*(-8*10**-4)-4.72*10**-20
            c=4.91*10**-14*(-8*10**-4)**2-3.46*10**-18*(-8*10**-4)+1.6*10**-20
            index=np.where(t==j)[0][0]
            pos=x[index]
            speed=v[index]
            E.append(a*pos**2+b*pos+c+0.5*6.64215627*10**(-26)*speed**2)
            T.append(0.5*6.64215627*10**(-26)*speed**2)
            U.append(a*pos**2+b*pos+c)
    plt.plot(10**6*x,T,label=f'v=-8/95*10**{i+1}')



plt.xlabel('time')
plt.ylabel('the position of the ion')
plt.legend()
plt.show()



plt.plot(pos,b)
plt.xlabel('position/m')
plt.ylabel('coefficient1')
plt.show()
'''

'''
plt.plot(pos,fre)
plt.xlabel('position')
plt.ylabel('frequency')
plt.show()


def delta_pos(delta,pos):
    coeffs_2=np.polyfit(pos,delta,2)
    linear=np.poly1d(coeffs_2)
    delta_fit=linear(pos)
    r2=r2_score(delta,delta_fit)
    print(r2)
    print(coeffs_2)
    plt.plot(pos,delta,color='r',label='original')
    plt.plot(pos,delta_fit,color='b',label='predicted')
    plt.xlabel('position/m')
    plt.ylabel('$delta$')
    plt.legend()
    plt.show()
    return coeffs_2
delta_pos(delta,pos)


def fre_pos(pos,fre):
    coeffs_3=np.polyfit(pos,fre,2)
    poly_2=np.poly1d(coeffs_3)
    fre_fit=poly_2(pos)
    r2=r2_score(fre,fre_fit)
    print(r2)
    print(coeffs_3)
    plt.plot(pos,fre,color='r',label='original')
    plt.plot(pos,fre_fit,color='b',label='predicted')
    plt.xlabel('position/m')
    plt.ylabel('b')
    plt.legend()
    plt.show()
    return

fre_pos(pos,b)

def c_fit(pos,c):
    coeffs_4=np.polyfit(pos,c,2)
    poly_3=np.poly1d(coeffs_4)
    c_fit=poly_3(pos)
    r2=r2_score(c,c_fit)
    print(r2)
    print(coeffs_4)
    plt.plot(pos,c,color='r',label='original')
    plt.plot(pos,c_fit,color='b',label='predicted')
    plt.xlabel('position/m')
    plt.ylabel('constant coefficient')
    plt.legend()
    plt.show()
    return
c_fit(pos,c)
 '''   
