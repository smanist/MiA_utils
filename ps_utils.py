"""
@package ps_utils
@author Dr. Daning Huang

Collection of utilities for examples and visualizations for
PySINDy (https://pysindy.readthedocs.io/en/latest/examples/index.html).

Developed for AERSP 597 classes

@date 04/20/2022
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps

integrator_keywords = {
    'rtol'   : 1e-12,
    'method' : 'LSODA',
    'atol'   : 1e-12
}

# ----------------------------------------------
# Example system
# ----------------------------------------------
def fSysNLn(t, y, om=3, rr=1):
    _r = np.linalg.norm(y)
    _po = -_r**2 * (_r**2-rr)
#     _po = -_r * (_r-rr)
    _f1 =-om*y[1] + y[0] * _po
    _f2 = om*y[0] + y[1] * _po
    _f  = np.array([_f1, _f2])
    return _f

fr = lambda x, y: (x**2+y**2)
clib_func = [
    lambda x: x,
    lambda x, y: x * fr(x,y),
    lambda x, y: y * fr(x,y),
    lambda x, y: x * fr(x,y)**2,
    lambda x, y: y * fr(x,y)**2]

clib_name = [
    lambda x : x,
    lambda x, y: f'{x}r^2',
    lambda x, y: f'{y}r^2',
    lambda x, y: f'{x}r^4',
    lambda x, y: f'{y}r^4']

clib_lco = ps.CustomLibrary(
    library_functions=clib_func,
    function_names=clib_name)

# ----------------------------------------------
# Data generation
# ----------------------------------------------
def genDataFixPar(dt, Tf, eps, x0s, arg):
    _t = np.arange(0, Tf, dt)
    _t_span = (_t[0], _t[-1])

    np.random.seed(42)  # A hack to enforce data reproducibility

    _xs = []
    for _x0 in x0s:
        _sol = solve_ivp(fSysNLn, _t_span, _x0,
                         args=arg, t_eval=_t, **integrator_keywords)
        _res = _sol.y.T
        if eps > 0:
            _res += np.random.rand(*_res.shape) * eps
        _xs.append(_res)
    return _t, _xs

def genDataMultiPar(dt, Tf, eps, x0s, args, aIdx):
    _dat = []
    for _arg in args:
        _t, _xs = genDataFixPar(dt, Tf, eps, x0s, _arg)
        _ones = np.ones((len(_t),1)) * np.array([_arg[aIdx]]).reshape(1,-1)
        _tmp = [np.hstack([_x,_ones]) for _x in _xs]
        _dat += _tmp
    return _t, _dat

# ----------------------------------------------
# Visualization
# ----------------------------------------------
def pltQuiver(rng, num, func, rr=None, ax=None):
    x0, x1 = rng
    xs = np.linspace(x0,x1,num)
    X,Y = np.meshgrid(xs,xs)
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    for _i in range(num):
        for _j in range(num):
            U[_i,_j], V[_i,_j] = func([X[_i,_j],Y[_i,_j]])
    R = np.sqrt(U**2+V**2)
    R[R==0] = 1e-3
    U /= R
    V /= R

    if ax is None:
        ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
    else:
        plt.sca(ax)
    plt.quiver(X, Y, U, V, scale=20)
    plt.xlabel('x')
    plt.ylabel('y')
    if rr is not None:
        if rr > 0:
            th = np.linspace(0,2*np.pi,101)
            plt.plot(rr*np.cos(th), rr*np.sin(th), 'r--')
        else:
            plt.plot(0.0, 0.0, 'ro', markersize=4)

def cmpTraj2D(true, pred, t, Ndim, lbls=['Data', 'Prediction']):
    _Nsol = len(true)
    f, ax = plt.subplots(nrows=Ndim, sharex=True, figsize=(10,3*Ndim))
    for _j in range(Ndim):
        for _i in range(_Nsol):
            _l1, = ax[_j].plot(t, true[_i][:,_j], 'b-', label=lbls[0])
            if pred is not None:
                _l2, = ax[_j].plot(t, pred[_i][:,_j], 'r--', label=lbls[1])
        ax[_j].set_ylabel(f'x{_j}')
    ax[-1].set_xlabel('Time')
    if pred is not None:
        plt.legend(handles=[_l1,_l2])

def cmpTrajPhase(true, pred=None, lbls=['Data', 'Prediction']):
    _Nsol = len(true)
    f = plt.figure(figsize=(8,6))
    for _i in range(_Nsol):
        _l1, = plt.plot(true[_i][:,0], true[_i][:,1], 'b-', label=lbls[0])
        if pred is not None:
            _l2, = plt.plot(pred[_i][:,0], pred[_i][:,1], 'r--', label=lbls[1])
    plt.xlabel('x')
    plt.ylabel('y')
    if pred is not None:
        plt.legend(handles=[_l1,_l2])

def cmpTraj3D(true, pred, Ipar):
    f = plt.figure(figsize=(12,5))
    a1 = f.add_subplot(121, projection="3d")
    a2 = f.add_subplot(122, projection="3d")
    for _sol, _ax in [(true,a1), (pred,a2)]:
        for _s in _sol:
            _ax.plot(_s[:,Ipar], _s[:,0], _s[:,1], '-')
        _ax.set_xlabel('Parameter')
        _ax.set_ylabel('x0')
        _ax.set_zlabel('x1')
    a1.set_title('Truth')
    a2.set_title('Prediction')

def cmpPhasePlane(args, mods, ttls):
    _Nf = len(mods)
    _rr = np.sqrt(args[1])
    _f1 = lambda x: fSysNLn(0, x, om=args[0], rr=args[1])
    _fs = [lambda x, mod=_m: mod.predict(np.array(x).reshape(1,-1)).reshape(-1) for _m in mods]
    f, ax = plt.subplots(ncols=_Nf+1, sharey=True, figsize=(2+5*(_Nf+1),5))
    pltQuiver([-3,3], 25, _f1, rr=_rr, ax=ax[0])
    for _i in range(_Nf):
        pltQuiver([-3,3], 25, _fs[_i], rr=_rr, ax=ax[_i+1])
        ax[_i+1].set_title(ttls[_i])
    ax[0].set_title('Truth')

# ----------------------------------------------
# Manipulation of SINDy objects
# ----------------------------------------------
def predTrajFixPar(model, x0s, t):
    _sol = []
    for _x0 in x0s:
        print(f'Solving {_x0}')
        _sol.append(model.simulate(_x0, t))
    return _sol

def predTrajMultiPar(model, x0s, t, args):
    _sol = []
    for _arg in args:
        _x0s = [np.hstack([_x0, np.array(_arg)]) for _x0 in x0s]
        _sol += predTrajFixPar(model, _x0s, t)
    return _sol

def wsWrapper(model, lib, opt, xs, ts):
    _m = ps.SINDy(feature_library=lib, optimizer=opt, feature_names=["x", "y"])
    _m.fit(xs, ts, multiple_trajectories=True)
    _m.model.steps[-1][1].optimizer.coef_ = model.coefficients()
    return _m