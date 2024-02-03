import numpy as np 
import matplotlib.pyplot as plt
import random
import math

f_x = lambda x: 1/4 * x**4 - 1/2 * x**2 - 1/16 * x

def golden_proportion(a, b, c):
    step = (3 - 5**(1/2))/2
    b_a = abs(b-a)
    c_b = abs(c-b)
    if b_a > c_b:
        d = a + step * b_a
    else:
        d = b + step * c_b
    return d


def golden_search(start, end, fx, tolerance=1e-6,max_iter=1000):
    a = start
    c = end
    b = (a+c)/2
    
    convergance = []
    iter = 0
    while True:
        d = golden_proportion(a,b,c)
        f_d = fx(d)
        f_b = fx(b)
        old_b = b
        if f_d < f_b:
            if d < b:
                c = b
                b = d
            else:
                a = b
                b = d
        else:
            if d < b:
                a = d
            else:
                c = d
        a_c = abs(a - c)
        convergance.append(a_c)
        iter += 1
        if a_c < tolerance * (abs(old_b) + abs(d)):
            break
        if iter > max_iter:
            break
        
    return a, c, convergance, iter


def brent_method(a, b, c, f_a, f_b, f_c):
    a_times_fc_fb = a * (f_c - f_b)
    b_times_fa_fc = b * (f_a - f_c)
    c_times_fb_fa = c * (f_b - f_a)

    d = (a_times_fc_fb * a + b_times_fa_fc * b + c_times_fb_fa * c) / (a_times_fc_fb + b_times_fa_fc + c_times_fb_fa)
    d /= 2
    
    return d


def narrowing(a, b, c, d, fx):
    fun_b = fx(b)
    fun_d = fx(d)

    if fun_d < fun_b:
        if d < b:
            c = b
            b = d
        else:
            a = b
            b = d
    else:
        if d < b:
            a = d
        else:
            c = d

    return a, b, c
    
        

def brent_search(start, end, fx, tolerance=1e-6, max_iter=1000):
    a = start
    c = end
    b = (a+c)/2
    
    convergance = []
    iter = 0
    while True:
        a_c = abs(a - c)
        old_b = b
        
        d = brent_method(a, b, c, fx(a), fx(b), fx(c))
        a, b, c = narrowing(a, b, c, d, fx)
        
        if a >= d or d >= c or abs(a - c) >= a_c / 2:
            d = (a + c) / 2
            a, b, c = narrowing(a, b, c, d, fx)
        
        convergance.append(a_c)
        iter += 1
        if a_c < tolerance * (abs(old_b) + abs(d)):
            break
        if iter > max_iter:
            break
        
    return a, c, convergance, iter

if __name__ == "__main__":
        
    a = 0
    c = 2
    g1, g2, convergance_gs, iterations_gs = golden_search(a, c, f_x)
    print("\n#####################")
    print("Golden search method:\n    ",
        f"result:[{g1},{g2}]\n    ", 
        f"iteratotions: {iterations_gs}")
    print("#####################\n")
    
    b1, b2, convergance_bs, iterations_bs = brent_search(a, c, f_x)
    print("\n#####################")
    print("Brent search method:\n    ",
        f"result:[{b1},{b2}]\n    ", 
        f"iteratotions: {iterations_bs}")
    print("#####################\n")
    
    
    
    #visualize convergance_gs and iterations_gs
    plt.plot(convergance_gs, 'r-', label="Golden search method")
    plt.plot(convergance_bs, 'b-', label="Brent search method")
    plt.title("Convergance of Golden and Brent search methods")
    plt.xlabel("Iterations")
    plt.ylabel("Convergance")
    plt.grid(True)
    plt.legend()
    plt.show()