"""
@package ds_utils
@author Dr. Daning Huang

Collection of utilities for examples and visualizations for
dynamical systems.

Developed for AERSP 597 classes

@date 04/05/2022
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import scipy.integrate as si
import scipy.linalg as sl
import scipy.signal as ss
# https://github.com/andrenarchy/pseudopy for pseudospectra
from pseudopy import NonnormalAuto, NonnormalMeshgrid

# ----------------------------------------------
# Example system
# ----------------------------------------------

def sys2D(delta, lam=None):
    """
    This is a non-normal system.  As delta->0, non-normality increases.
    The example is taken from AIAAJ2017/2020.

    Explicitly, the system matrix is the following:
        if lam is None:
            _l = np.array([-0.1, -0.2])
        else:
            _l = np.array(lam)
        _t = np.pi/4
        _V = np.array([
            [np.cos(_t-delta), np.cos(_t+delta)],
            [np.sin(_t-delta), np.sin(_t+delta)]
        ])
        _Vi = np.array([
            [np.sin(_t+delta), -np.cos(_t+delta)],
            [-np.cos(_t+delta), np.sin(_t+delta)]
        ]) / np.sin(2*delta)
        return _V.dot(np.diag(_l)).dot(_Vi)
    """
    if lam is None:
        _l1, _l2 = -0.1, -0.2
    else:
        _l1, _l2 = lam
    _ls = _l1+_l2
    _dl = _l1-_l2
    _cs = 1/np.sin(2*delta)
    _ct = 1/np.tan(2*delta)
    _A = 0.5*np.array([
        [_ls+_dl*_cs, -_dl*_ct],
        [_dl*_ct, _ls-_dl*_cs]])
    return _A

def makeSys2D(delta, lam=None):
    # Make a LTI system in Scipy
    _A = sys2D(delta, lam)
    _B, _C = np.eye(2), np.eye(2)
    _D = np.zeros((2,2))
    return ss.StateSpace(_A, _B, _C, _D)

def fSysNLn(t,y,delta=np.pi/4,lam=[-0.1,-0.2],rr=1):
    # For Scipy.integrate
    # We create a non-normal stable system at the center
    # and blend it with a limit cycle system with radius of 2
    # When rr<0, the limit cycle is unstable
    _A = sys2D(delta, lam)
    _r = np.linalg.norm(y)
    _e = np.exp(-10*(_r-1.8))
    _c = _e / (1+_e)
    _po = -_r**2 * (_r-2)
    _f1 = -y[1] + y[0] * _po
    _f2 = y[0] + y[1] * _po
    _f  = rr * np.array([_f1, _f2]) * (1-_c) + _A.dot(y) * _c
    return _f

def fSysLin(t,y,delta=np.pi/4,lam=[-0.1,-0.2]):
    # For Scipy.integrate
    # Same as makeSys2D
    return sys2D(delta, lam).dot(y)

# ----------------------------------------------
# Response visualization
# ----------------------------------------------
def pltPhasePlaneNL(xrng, xnum, yrng, ynum, delta, lms, rr, ifLin=True):
    tf = 50
    T  = np.linspace(0,tf,2001)
    x0, x1 = xrng
    y0, y1 = yrng
    xs = np.linspace(x0,x1,xnum)
    ys = np.linspace(y0,y1,ynum)
    X,Y = np.meshgrid(xs,ys)
    plt.figure(figsize=(10,10))
    for _i in range(ynum):
        for _j in range(xnum):
            _x, _y = X[_i,_j], Y[_i,_j]
            sol = si.solve_ivp(fSysNLn, [0,tf], [_x,_y], args=(delta, lms, rr), t_eval=T)
            plt.plot(_x, _y, 'ko', markerfacecolor='none')
            l1, = plt.plot(sol.y[0], sol.y[1], 'b-', alpha=0.5, label='Nonlinear')
            if ifLin:
                lin = si.solve_ivp(fSysLin, [0,tf], [_x,_y], args=(delta, lms), t_eval=T)
                l2, = plt.plot(lin.y[0], lin.y[1], 'r-', alpha=0.5, label='Linear')
    if ifLin:
        plt.legend(handles=[l1,l2])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

def pltQuiver(rng, num, delta, lms, rr):
    x0, x1 = rng
    xs = np.linspace(x0,x1,num)
    X,Y = np.meshgrid(xs,xs)
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    for _i in range(num):
        for _j in range(num):
            U[_i,_j], V[_i,_j] = fSysNLn(0, [X[_i,_j],Y[_i,_j]],
                                         delta=delta,lam=lms,rr=rr)
    R = np.sqrt(U**2+V**2)
    R[R==0] = 1e-3
    U /= R
    V /= R
    
    th = np.linspace(0,2*np.pi,101)

    plt.figure(figsize=(8,8))
    plt.quiver(X, Y, U, V, scale=20)
    plt.plot(2*np.cos(th), 2*np.sin(th), 'r--')

def _procArr(*args):
    _Ns = [len(_a) for _a in args]
    _m  = np.argmax(_Ns)
    _N  = _Ns[_m]
    _res = []
    for _i in range(len(args)):
        _a = args[_i]
        _res.append(_a if _Ns[_i]>1 else [_a[0]]*_N)
    return _res, args[_m]

def pltRespSweep(x0s, x1s, rs, delta, lms, label='Amp.', tf=50, ifLin=True):
    (_x0s, _x1s, _rs), _par = _procArr(x0s, x1s, rs)
    T  = np.linspace(0,tf,2001)
    _bs = []
    f, ax = plt.subplots(ncols=1,figsize=(12,5))
    for _i, _rr in enumerate(_rs):
        X0 = [_x0s[_i], _x1s[_i]]
        sol = si.solve_ivp(fSysNLn, [0,tf], X0, args=(delta,lms,_rr), t_eval=T)
        l1, = plt.plot(sol.t, sol.y[0], 'b-', alpha=0.5, label='Nonlinear')
        if ifLin:
            lin = si.solve_ivp(fSysLin, [0,tf], X0, args=(delta,lms), t_eval=T)
            l2, = plt.plot(lin.t, lin.y[0], 'r-', alpha=0.5, label='Linear')

        plt.text(sol.t[-1], sol.y[0,-1], f'{label}={_par[_i]:4.3f}')

        _d = sol.y[1][-400:]
        _m = np.max(np.abs(_d))
        if np.max(_d) > 1.2:
            _bs.append(np.max(_d))
        else:
            _bs.append(0)
    if ifLin:
        plt.legend(handles=[l1,l2])

    plt.xlabel('Time')
    plt.ylabel(label)

# ----------------------------------------------
def pltPhasePlaneSS(X0, ds):
    # Example trajectories showing non-normality
    T = np.linspace(0,100,201)
    U = 0
    f = plt.figure(figsize=(8,6))
    for _d in ds:
        S = makeSys2D(_d)
        _, _, X = ss.lsim(S, U, T, X0=X0)
        plt.plot(X[:,0], X[:,1])
        plt.text(X[20,0], X[20,1], f'$\delta$={_d:3.2f}')
    plt.plot(X0[0], X0[1], 'bo')
    plt.text(X0[0]+0.1, X0[1], 'Start')
    plt.plot(0, 0, 'rs')
    plt.text(-0.1, 0.1, 'End')
    th = np.linspace(0,np.pi/2,51)
    plt.plot(2*np.cos(th), 2*np.sin(th), 'k--')
    plt.text(0, 1.7, 'Approx. linear region')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

def pltEnergy(X0, ds, awid=0.2):
    # Viewing non-normality in terms of energy
    T = np.linspace(0,100,201)
    U = 0
    plt.figure(figsize=(8,6))
    for _d in ds:
        S = makeSys2D(_d)
        _, _, X = ss.lsim(S, U, T, X0=X0)
        E = X[:,0]**2+X[:,1]**2
        m = np.argmax(E)
        plt.plot(T, E)
        plt.text(T[20], E[20], f'$\delta$={_d:3.2f}')
        if _d < np.pi/4:
            plt.plot([T[m]-2, T[m]+2], [E[m], E[m]], 'r-', linewidth=5)
        plt.arrow(T[0], E[0], T[5]-T[0], E[5]-E[0], width=awid, edgecolor='none', facecolor='k')
    plt.xlim([-5, 40])
    plt.xlabel('Time')
    plt.ylabel('Energy: $x_1^2+x_2^2$')


# ----------------------------------------------
# MIMO analysis
# ----------------------------------------------
def fftProc(T, U, Y):
    N  = len(T)
    dt = T[1]-T[0]
    ff = fftfreq(N, dt)[:N//2] * (2*np.pi)
    Uf = 2.0/N * np.abs(fft(U))[0:N//2]
    Yf = 2.0/N * np.abs(fft(Y))[0:N//2]
    return ff, Uf, Yf

def frMIMO(sys, w, ifFull=False):
    # Frequency sweep to obtain the gain characteristics of a MIMO system
    N = len(w)
    M = sys.A.shape[0]
    G = np.zeros((N,M))
    out, inp =[], []
    I = 1j*np.eye(M)
    for _i in range(N):
        try:
            AB = np.linalg.solve(w[_i]*I-sys.A, sys.B)
        except:
            AB = np.linalg.pinv(w[_i]*I-sys.A).dot(sys.B)
        u, s, vh = np.linalg.svd(sys.C.dot(AB) + sys.D)
        G[_i] = s
        if ifFull:
            # Row-wise storage
            out.append(u.T)
            inp.append(np.conj(vh))
    if ifFull:
        return np.squeeze(out), np.squeeze(G), np.squeeze(inp)
    return G

def genInput(mode, wT):
    # Generate input to a SSM given mode and omega*T
    # The mode is complex and gives the amplitude and phase
    _T = wT.reshape(-1,1)
    _M = mode.reshape(1,-1)
    return np.real(np.exp(1j*_T) * _M)

def extractMode(res, mode):
    # Project the response onto the given possibly complex mode
    return res.dot(sl.pinv(np.atleast_2d(mode))).reshape(-1)

def pltGainResp(sys, om, UGV, idx, Tf=80*np.pi):
    # Generate the SSM response given the input mode
    # Verify the gain from time and frequency responses
    out, gain, inp = UGV
    _T = np.linspace(0,Tf,4001)
    _U = genInput(inp[idx], om*_T)
    _, _Y, _ = ss.lsim(sys, _U, _T)
    _Uv = _U[:,0]
    _Yu = _Y[:,0]
    _scl = np.max(_Uv)
    _Uv /= _scl
    _Yu /= _scl

    _Te = 2000
    _f, _Uf, _Yf = fftProc(_T[_Te:], _Uv[_Te:], _Yu[_Te:])

    f, ax = plt.subplots(ncols=2, figsize=(12,3))
    ax[0].plot(_T, _Uv, 'r-')
    ax[0].plot(_T, _Yu, 'b--')
    ax[0].set_xlabel('Time, s')
    ax[0].set_ylabel('Response (real)')

    ax[1].plot(_f, _Uf, 'r-')
    ax[1].plot(_f, _Yf, 'b--')
    ax[1].set_xlim([0,3])
    ax[1].set_xlabel('Freq, rad')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_title(f'Num: {np.max(_Yf)/np.max(_Uf):5.4f}, Exact: {gain[idx]:5.4f}')

def pltSVDsweep(W, ds):
    st = ['b', 'r']
    plt.figure(figsize=(8,6))
    for _i, _d in enumerate(ds):
        S = makeSys2D(_d)
        G = frMIMO(S, W)
        plt.plot(W, G[:,0], st[_i]+'-', label=f'$\delta=${_d:4.3f}')
        plt.plot(W, G[:,1], st[_i]+'--')
    plt.plot([0.3, 0.3], [1, 50], 'k:')
    plt.legend()
    plt.xlabel('Freq, rad')
    plt.ylabel('Gain')


# ----------------------------------------------
# Non-normal analysis
# ----------------------------------------------
find = lambda condition: np.nonzero(np.ravel(condition))[0]

def numRange(A, resolution=0.01):
    # Algorithm from
    #     C. Cowen and E. Harel, An Effective Algorithm for Computing the Numerical Range, 1995
    # Initial Python implementation found by @nicoguaro
    A = np.asmatrix(A)
    e0, _ = np.linalg.eig(A)
    th = np.arange(0, 2*np.pi + resolution, resolution)
    w = []
    for j in th:
        Ath = np.exp(-1j*j)*A
        Hth = (Ath + Ath.H)/2
        e,r = np.linalg.eigh(Hth)
        r = np.matrix(r)
        e = np.real(e)
        s = find(e == e.max())
        if np.size(s) == 1:
            w.append(np.matrix.item(r[:,s].H*A*r[:,s]))
        else:
            Kth = 1j*(Hth - Ath)
            pKp = r[:,s].H*Kth*r[:,s]
            ee,rr = np.linalg.eigh(pKp)
            rr = np.matrix(rr)
            ee = np.real(ee)
            sm = find(ee == ee.min())
            temp = rr[:,sm[0]].H*r[:,s].H*A*r[:,s]*rr[:,sm[0]]
            w.append(temp[0,0])
            sM = find(ee == ee.max())
            temp = rr[:,sM[0]].H*r[:,s].H*A*r[:,s]*rr[:,sM[0]]
            w.append(temp[0,0])
    return np.array(w), e0

def pltNumRange(ds):
    plt.figure(figsize=(8,6))
    for _d in ds:
        _A = sys2D(_d)
        _w, _e = numRange(_A, resolution=0.01)
        plt.plot(_w.real, _w.imag)
        plt.plot(_e.real, _e.imag, 'ro')
        plt.plot(_w.real[0], _w.imag[0], 'ks')
        plt.text(_w.real[0], _w.imag[0]-0.05, f'Re={_w.real[0]:3.2f}')
        plt.text(_w.real[60], _w.imag[60], f'$\delta$={_d:3.2f}')
    plt.plot([0,0], [-2,2], 'k--')
    plt.xlim([-0.4, 0.4])
    plt.ylim([-0.5, 0.5])
    plt.xlabel('Real')
    plt.ylabel('Imag')


# ----------------------------------------------
def pltInitGain(delta, ti):
    T = np.linspace(0,100,201)
    U = 0
    S = makeSys2D(delta)
    tg = T[ti]
    H = sl.expm(S.A*tg)
    u, s, vh = np.linalg.svd(H)
    # print(np.hstack([u,np.diag(s),np.conj(vh.T)]))
    us = u.dot(np.diag(s)).T
    f, ax = plt.subplots(ncols=2, figsize=(12,5))
    for _i, _x0 in enumerate(vh):
        _, _, X = ss.lsim(S, U, T, X0=np.conj(_x0))
        E = np.linalg.norm(X, axis=1)
        t = f'Gain={s[_i]:4.3f}'
        ax[0].plot(X[:,0], X[:,1])
        ax[0].plot(X[ti,0], X[ti,1], 'bo')
        ax[0].plot(_x0[0], _x0[1], 'bo', markerfacecolor='none')
        ax[0].plot(us[_i,0], us[_i,1], 'bo')
        ax[0].text(us[_i,0]+0.1, us[_i,1], t)
        ax[1].plot(T, E)
        ax[1].plot(tg, s[_i], 'bo')
        ax[1].text(tg+2, s[_i], t)
    ax[0].set_xlabel('$x_1$')
    ax[0].set_ylabel('$x_2$')
    ax[0].set_title(f'Time={tg:3.2f}')
    ax[1].plot([tg, tg], s, 'k--')
    ax[1].set_xlim([-5, 40])
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('SQRT of Energy')


# ----------------------------------------------
def estKreiss(A):
    pA = NonnormalMeshgrid(A,
                           real_min=0.01, real_max=1, real_n=200,
                           imag_min=0, imag_max=0, imag_n=1,
                           method='svd')
    return np.max(pA.Real/pA.Vals)

def pltGainSweep(delta, ts):
    T = np.linspace(0,100,201)
    U = 0
    S = makeSys2D(delta)
    K = estKreiss(S.A)
    f = plt.figure(figsize=(8,6))
    Gmx = 0
    for _ti in ts:
        tg = T[_ti]
        H = sl.expm(S.A*tg)
        _, s, vh = np.linalg.svd(H)
        _, _, X = ss.lsim(S, U, T, X0=np.conj(vh[0]))
        E = np.linalg.norm(X, axis=1)
        Gmx = max(Gmx, s[0])
        plt.plot(T, E, 'b-')
    plt.plot([0, 40], [Gmx, Gmx], 'k--')
    plt.text(20, Gmx-0.1, 'Approx. Max Gain')
    plt.plot([0, 40], [K, K], 'k--')
    plt.text(20, K-0.1, 'Approx. Kreiss const.')
    plt.xlim([-5, 40])
    plt.xlabel('Time')
    plt.ylabel('SQRT of Energy')


# ----------------------------------------------
def pltPseudoSpec(ds, ks):
    f, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12,8))
    ax = axs.flatten()
    for _i, _d in enumerate(ds):
        A = sys2D(_d)
        pA = NonnormalAuto(A, 1e-5, 1e-1)
        plt.sca(ax[_i])
        pA.plot([10**k for k in ks], spectrum=np.linalg.eigvals(A))
        ax[_i].plot([0,0], [-0.3,0.3], 'k--')
        ax[_i].set_title(f'$\delta$={_d:4.3f}')