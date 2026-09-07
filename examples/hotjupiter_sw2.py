#!/usr/bin/env python3
"""热木星**非线性**旋转浅水（β-平面通道）昼夜环流求解器 (纯 Python + numpy).

与 `hotjupiter_sw.py`（线性 Matsuno–Gill）同物理谱系、同第一性原理强迫，
差异在本文件保留**完整非线性平流/动量通量项**，靠赤道波（东传 Kelvin + 西传
Rossby）的涡动动量通量在赤道上泵出**东向超自转喷射**——线性模型只有 ~0.1 m/s 的
jet，正是这里要超越的本源（Showman & Polvani 2011 赤道超自转标准结果）。

数值方法（极限模式纪律）：
  - **守恒有限体积 + Rusanov 通量**（内禀数值扩散），x 周期、y 通道自由滑移壁
    （v=0），这是强超自转喷射下最稳健的第一性原理谱系以外选择（任务允许的守恒保底）。
  - 同时保留一套伪谱(2/3 反混淆+超粘性)实现于 `NonlinearSWChannels`，供高阶对照。
  - 显式 RK2 + CFL 检查；发散→抛异常；单位 SI。
  - 守恒诊断：质量/能量收支闭合（终态窗口时间平均）。

方程（h 总层厚；U=hu, V=hv 为动量密度；f=βy；c²=g'·H₀）：
  ∂h/∂t + ∂(U)/∂x + ∂(V)/∂y = −(h − h_eq)/τ_rad
  ∂U/∂t + ∂(U²/h + g'h²/2)/∂x + ∂(UV/h)/∂y = f V − U/τ_drag
  ∂V/∂t + ∂(UV/h)/∂x + ∂(V²/h + g'h²/2)/∂y = −f U − V/τ_drag
h_eq 由 `hotjupiter_sw` 的第一性原理平衡层厚场给出（仅 x 昼夜不对称、y 对称）。
"""
from __future__ import annotations

import math

import numpy as np

import hotjupiter_sw as hj

PI = math.pi


class NonlinearSWChannels:
    """β-平面通道上非线性旋转浅水谱求解器。"""

    def __init__(self, name: str = "WASP-43b", nx: int = 64, ny: int = 32,
                 y_half_deg: float = 30.0, tau_rad: float | None = None,
                 tau_drag: float | None = None, nu4: float | None = None,
                 cfl: float | None = None, amp: float | None = None,
                 night_frac: float | None = None,
                 dealias_factor: float = 2.0 / 3.0):
        self.sys = hj.real_systems()[name]
        self.name = name
        self.nx, self.ny = nx, ny
        R = self.sys.planet_radius_m
        self.Lx = 2.0 * PI * R
        self.Ly = 2.0 * (y_half_deg * PI / 180.0) * R
        self.dx = self.Lx / nx
        self.dy = self.Ly / ny

        m = hj.MODEL
        self.gp = m["g_prime"]                      # 约化重力 [m/s²]
        self.H0 = m["H0"]                            # 参考层厚 [m]
        self.c2 = self.gp * self.H0                  # c² [m²/s²]
        self.tau_rad = m["tau_rad"] if tau_rad is None else tau_rad
        self.tau_drag = m["tau_drag"] if tau_drag is None else tau_drag
        self.amp = m["amp"] if amp is None else amp
        self.night_frac = m["night_frac"] if night_frac is None else night_frac
        self.cfl = m["cfl"] if cfl is None else cfl
        self.dealias = dealias_factor

        # ---- 物理网格（x 周期、y 通道半错列点）----
        self.xp = np.arange(nx) * self.dx            # x 坐标（FFT 只用间距）
        self.yp = (np.arange(ny) + 0.5) * self.dy - self.Ly / 2.0
        self.lons_rad = [(i + 0.5) * (2 * PI / nx) - PI for i in range(nx)]
        self.lats_rad = [self.yp[j] / R for j in range(ny)]
        self.f = (self.sys.beta * self.yp)[:, None]  # f(j,1)

        # ---- y 方向基（预存正交基与逆）----
        th = (np.arange(ny) + 0.5) * PI / ny
        self.kc = np.arange(ny)                      # cos 模 0..ny-1
        self.ks = np.arange(1, ny + 1)               # sin 模 1..ny
        self.C = np.cos(self.kc[np.newaxis, :] * th[:, np.newaxis])
        self.invC = np.linalg.inv(self.C)
        self.S = np.sin(self.ks[np.newaxis, :] * th[:, np.newaxis])
        self.invS = np.linalg.inv(self.S)
        self.kpi = PI / self.Ly                      # [1/m]

        # ---- x 方向波矢 (rfft) ----
        self.kx = 2.0 * PI * np.arange(nx // 2 + 1) / self.Lx
        # ---- 2/3 反混淆截断 ----
        # x：FFT Nyquist = nx/2，保留 ≤ dealias·(nx/2)（标准 2/3）
        self.Rx = int(self.dealias * (nx // 2))
        # y：半错列 DCT 上同样取 2/3 Nyquist
        self.Rc = int(self.dealias * (ny // 2))

        # ---- 第一性原理强迫场（复用 hotjupiter_sw 的平衡层厚）----
        heq_list, self.t_sub = hj.equilibrium_height_field(
            self.sys, nx, ny, self.lons_rad, self.lats_rad,
            self.night_frac, self.amp)
        self.h_eq = np.asarray(heq_list, dtype=float)

        # ---- 超粘性系数 ν₄∇⁴ (网格尺度耗散；不显著限制 dt) ----
        self.nu4 = 5.0e20 if nu4 is None else nu4

        # ---- 初态：静止基态 h=H0, u=v=0 ----
        self.h = np.full((ny, nx), self.H0, dtype=float)
        self.u = np.zeros((ny, nx), dtype=float)
        self.v = np.zeros((ny, nx), dtype=float)

    # ---------- 谱导数算子 ----------
    def ddx(self, f):
        """∂/∂x（周期 Fourier；返回 f 的同 y 奇偶性）。"""
        F = np.fft.rfft(f, axis=1)
        F *= (1j * self.kx)
        F[:, self.nx // 2] = 0.0
        return np.fft.irfft(F, n=self.nx, axis=1)

    def dxx(self, f):
        F = np.fft.rfft(f, axis=1)
        F *= -(self.kx ** 2)
        return np.fft.irfft(F, n=self.nx, axis=1)

    def dy_even2odd(self, p):
        """∂_y：cos 基(偶数场 u,h) → sin 基(奇数场 v)。"""
        c = self.invC @ p
        sc = np.zeros_like(c)
        sc[:-1] = -(self.ks[:-1] * self.kpi)[:, None] * c[1:]
        return self.S @ sc

    def dy_odd2even(self, p):
        """∂_y：sin 基(奇数场 v) → cos 基(偶数场)。"""
        sv = self.invS @ p
        cc = np.zeros_like(sv)
        cc[1:] = (self.ks[:-1] * self.kpi)[:, None] * sv[:-1]
        return self.C @ cc

    def dyy_even(self, p):
        c = self.invC @ p
        return self.C @ (c * (-(self.kc * self.kpi) ** 2)[:, None])

    def dyy_odd(self, p):
        sv = self.invS @ p
        return self.S @ (sv * (-(self.ks * self.kpi) ** 2)[:, None])

    def bih_even(self, p):
        L = self.dxx(p) + self.dyy_even(p)
        return self.dxx(L) + self.dyy_even(L)

    def bih_odd(self, p):
        L = self.dxx(p) + self.dyy_odd(p)
        return self.dxx(L) + self.dyy_odd(L)

    # ---------- 2/3 反混淆滤波器 ----------
    def _choppy(self, f):
        F = np.fft.rfft(f, axis=1)
        F[:, self.Rx + 1:] = 0.0
        f[...] = np.fft.irfft(F, n=self.nx, axis=1)

    def _chop_even(self, f):
        self._choppy(f)
        c = self.invC @ f
        c[self.Rc + 1:] = 0.0
        f[...] = self.C @ c

    def _chop_odd(self, f):
        self._choppy(f)
        sv = self.invS @ f
        sv[self.Rc:] = 0.0
        f[...] = self.S @ sv

    # ---------- 右端项 ----
    def rhs(self, h, u, v):
        hu = h * u
        hv = h * v
        # 动量平流
        adv_u = u * self.ddx(u) + v * self.dy_even2odd(u)
        adv_v = u * self.ddx(v) + v * self.dy_odd2even(v)
        # 质量通量散度
        div_mass = self.ddx(hu) + self.dy_odd2even(hv)
        # 气压梯度
        pgr_x = self.gp * self.ddx(h)
        pgr_y = self.gp * self.dy_even2odd(h)
        # 右端项完成（物理项反混淆后再叠加超粘性）
        du = -adv_u + self.f * v - pgr_x - u / self.tau_drag
        dv = -adv_v - self.f * u - pgr_y - v / self.tau_drag
        dh = -div_mass - (h - self.h_eq) / self.tau_rad
        self._chop_even(du)
        self._chop_odd(dv)
        self._chop_even(dh)
        du = du + self.nu4 * self.bih_even(u)
        dv = dv + self.nu4 * self.bih_odd(v)
        dh = dh + self.nu4 * self.bih_even(h)
        self._chop_even(du)
        self._chop_odd(dv)
        self._chop_even(dh)
        return du, dv, dh

    # ---------- 时间步 ----------
    def cfl_timestep(self):
        c = math.sqrt(self.c2)
        dxm = min(self.dx, self.dy)
        umax = float(np.max(np.abs(self.u)))
        dt_adv = self.cfl * dxm / (c + umax + 1e-9)
        # 超粘性显式稳定：dt·ν₄·(k²max)² ≤ ~2.8 (RK4), k²max 取 dealias 截断处的最大波数平方
        k2max = (2 * PI * self.Rx / self.Lx) ** 2 + (PI * self.Rc / self.Ly) ** 2
        dt_hyp = 2.8 / (self.nu4 * k2max ** 2 + 1e-300)
        dt_force = 0.1 * self.tau_rad
        return min(dt_adv, dt_hyp, dt_force)

    def energy_density(self, h, u, v):
        ke = np.mean(0.5 * (u * u + v * v))
        pe = np.mean(0.5 * self.gp * (h - self.H0) ** 2 / self.H0)
        return ke + pe

    # ---------- 积分 ----------
    def integrate(self, t_end_s: float) -> dict:
        h, u, v = self.h, self.u, self.v
        dt = self.cfl_timestep()
        t = 0.0
        n = 0
        t_hist = [t]
        e_hist = [float(self.energy_density(h, u, v))]
        m_hist = [float(np.mean(h))]
        while t < t_end_s - 1e-9:
            dt = self.cfl_timestep()
            dt_now = min(dt, t_end_s - t)
            du0, dv0, dh0 = self.rhs(h, u, v)
            # RK4
            k1u, k1v, k1h = du0, dv0, dh0
            h2 = h + 0.5 * dt_now * k1h
            u2 = u + 0.5 * dt_now * k1u
            v2 = v + 0.5 * dt_now * k1v
            k2u, k2v, k2h = self.rhs(h2, u2, v2)
            h3 = h + 0.5 * dt_now * k2h
            u3 = u + 0.5 * dt_now * k2u
            v3 = v + 0.5 * dt_now * k2v
            k3u, k3v, k3h = self.rhs(h3, u3, v3)
            h4 = h + dt_now * k3h
            u4 = u + dt_now * k3u
            v4 = v + dt_now * k3v
            k4u, k4v, k4h = self.rhs(h4, u4, v4)
            h += (dt_now / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)
            u += (dt_now / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
            v += (dt_now / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            t += dt_now
            n += 1
            if not (np.isfinite(h).all() and np.isfinite(u).all()
                    and np.isfinite(v).all()):
                raise FloatingPointError(
                    f"非线性谱模型发散 @ t={t:.3e}s n={n}")
            if n > 2_000_000:
                raise RuntimeError("积分步骤异常过多，疑似发散")
            t_hist.append(t)
            e_hist.append(float(self.energy_density(h, u, v)))
            m_hist.append(float(np.mean(h)))

        # ---- 收支闭合：终态窗口时间平均 ----
        t_arr = np.asarray(t_hist)
        e_arr = np.asarray(e_hist)
        m_arr = np.asarray(m_hist)
        w = self.tau_rad
        ma = np.abs(t_arr - t_arr[-1]) < w
        mb = (np.abs(t_arr - t_arr[-1]) > w) & (np.abs(t_arr - t_arr[-1]) < 2 * w)
        E_late = float(np.mean(e_arr[ma]))
        E_prior = float(np.mean(e_arr[mb]))
        rel_energy_drift = (E_late - E_prior) / E_prior if E_prior > 1e-30 else 0.0
        mass_late = float(np.mean(m_arr[ma]))
        mass_drift = (mass_late - self.H0) / self.H0

        return {
            "name": self.name,
            "t_end_s": t,
            "steps": n,
            "dt_s": dt,
            "CFL": self.cfl,
            "mass_drift": mass_drift,
            "rel_energy_drift": rel_energy_drift,
            "t_sub": self.t_sub,
            "nu4": self.nu4,
        }

    # ---------- 从层厚恢复温度 + 诊断 ----------
    def temperature(self):
        return self.t_sub * (1.0 + (self.h / self.H0 - 1.0) / self.amp)

    def analyze(self) -> dict:
        T = self.temperature()
        u = self.u
        nx, ny = self.nx, self.ny
        R = self.sys.planet_radius_m
        deg = 180.0 / PI
        eq_rows = [j for j in range(ny) if abs(self.yp[j]) < (15.0 * PI / 180.0) * R]
        day_idx = [i for i in range(nx) if abs(self.lons_rad[i]) < PI / 2.0]
        night_idx = [i for i in range(nx) if abs(self.lons_rad[i]) >= PI / 2.0]
        dayT = float(np.mean([[T[j, i] for i in day_idx] for j in eq_rows]))
        nightT = float(np.mean([[T[j, i] for i in night_idx] for j in eq_rows]))
        T_merid = np.mean(T[eq_rows, :], axis=0)
        imax = int(np.argmax(T_merid))
        offset_deg = self.lons_rad[imax] * deg
        u_zonal = np.mean(u[eq_rows, :], axis=1)
        jet_max = float(np.max(u_zonal))
        return {
            "day_mean_K": round(dayT, 1),
            "night_mean_K": round(nightT, 1),
            "day_night_delta_T_K": round(dayT - nightT, 1),
            "hot_spot_offset_deg": round(offset_deg, 1),
            "equatorial_jet_max_ms": round(jet_max, 1),
        }


# ════════════ 守恒有限体积 + Rusanov（工作主力）════════════
class NonlinearSWChannelsFVM:
    """β-平面通道非线性旋转浅水，守恒有限体积 + Rusanov 通量。

    状态：总层厚 h（=H），动量密度 U=h·u, V=h·v（格点胞平均）。x 周期、y 通道，
    墙边界自由滑移（v=0 ⇒ Fy=0 at 南北墙）。Rusanov 通量提供每步数值扩散，
    令强超自转喷射在粗网格上也能稳定平衡而不发生谱伪谱那种能量级联发散。
    """

    def __init__(self, name: str = "WASP-43b", nx: int = 64, ny: int = 32,
                 y_half_deg: float = 30.0, tau_rad: float | None = None,
                 tau_drag: float | None = None, nu4: float | None = None,
                 cfl: float | None = None, amp: float | None = None,
                 night_frac: float | None = None):
        self.sys = hj.real_systems()[name]
        self.name = name
        self.nx, self.ny = nx, ny
        R = self.sys.planet_radius_m
        self.Lx = 2.0 * PI * R
        self.Ly = 2.0 * (y_half_deg * PI / 180.0) * R
        self.dx = self.Lx / nx
        self.dy = self.Ly / ny

        m = hj.MODEL
        self.gp = m["g_prime"]
        self.H0 = m["H0"]
        self.c2 = self.gp * self.H0
        self.tau_rad = m["tau_rad"] if tau_rad is None else tau_rad
        self.tau_drag = m["tau_drag"] if tau_drag is None else tau_drag
        self.amp = m["amp"] if amp is None else amp
        self.night_frac = m["night_frac"] if night_frac is None else night_frac
        self.cfl = m["cfl"] if cfl is None else cfl
        self.nu4 = 2.0e20 if nu4 is None else nu4

        # 胞心网格：x 周期、y 通道 ±(y_half_deg)
        self.xp = (np.arange(nx) + 0.5) * self.dx
        self.yp = -self.Ly / 2.0 + (np.arange(ny) + 0.5) * self.dy
        self.lons_rad = [(self.xp[i] / R) - PI for i in range(nx)]
        self.lats_rad = [self.yp[j] / R for j in range(ny)]
        self.f = (self.sys.beta * self.yp)[:, None]

        heq_list, self.t_sub = hj.equilibrium_height_field(
            self.sys, nx, ny, self.lons_rad, self.lats_rad,
            self.night_frac, self.amp)
        self.h_eq = np.asarray(heq_list, dtype=float)

        self.H = np.full((ny, nx), self.H0, dtype=float)
        self.U = np.zeros((ny, nx), dtype=float)
        self.V = np.zeros((ny, nx), dtype=float)

    # ---- minmod 斜率限制（二阶 MUSCL，σ=0→一阶，用少数值扩散却保单调）----
    def _slope_minmod(self, q):
        dl = q - np.roll(q, 1, axis=1)
        dr = np.roll(q, -1, axis=1) - q
        s = np.where(dl * dr > 0, np.sign(dl) * np.minimum(np.abs(dl), np.abs(dr)), 0.0)
        return s

    def _slope_minmod_y(self, q):
        dl = q.copy(); dl[1:] = q[1:] - q[:-1]; dl[0] = 0.0
        dr = q.copy(); dr[:-1] = q[1:] - q[:-1]; dr[-1] = 0.0
        s = np.where(dl * dr > 0, np.sign(dl) * np.minimum(np.abs(dl), np.abs(dr)), 0.0)
        s[0, :] = 0.0
        s[-1, :] = 0.0
        return s

    # ---- HLLC 通量（低数值扩散，正常量在 normal 方向）----
    def _hllc(self, hL, mnL, mtL, hR, mnR, mtR):
        gp = self.gp
        uL = mnL / hL; uR = mnR / hR
        cL = np.sqrt(gp * hL); cR = np.sqrt(gp * hR)
        slr = np.sqrt(hL); srr = np.sqrt(hR); den = slr + srr + 1e-30
        ubar = (slr * uL + srr * uR) / den
        cbar = np.sqrt(0.5 * gp * (hL + hR))
        pL = 0.5 * gp * hL * hL; pR = 0.5 * gp * hR * hR
        SL = np.minimum(uL - cL, ubar - cbar)
        SR = np.maximum(uR + cR, ubar + cbar)
        DL = hL * (SL - uL) - hR * (SR - uR) + 1e-30
        Sstar = (pR - pL + hL * uL * (SL - uL) - hR * uR * (SR - uR)) / DL
        # 星区左/右守恒态
        aL = SL - uL; bL = SL - Sstar + 1e-30
        UstL_h = hL * aL / bL
        UstL_mn = Sstar * UstL_h
        UstL_mt = mtL * aL / bL
        aR = SR - uR; bR = SR - Sstar + 1e-30
        UstR_h = hR * aR / bR
        UstR_mn = Sstar * UstR_h
        UstR_mt = mtR * aR / bR
        # 左右通量
        FL_h = mnL; FL_mn = mnL * uL + pL; FL_mt = mtL * uL
        FR_h = mnR; FR_mn = mnR * uR + pR; FR_mt = mtR * uR
        # 星区通量 = 物理通量 + S·(U*−U)
        FstL_h = FL_h + SL * (UstL_h - hL)
        FstL_mn = FL_mn + SL * (UstL_mn - mnL)
        FstL_mt = FL_mt + SL * (UstL_mt - mtL)
        FstR_h = FR_h + SR * (UstR_h - hR)
        FstR_mn = FR_mn + SR * (UstR_mn - mnR)
        FstR_mt = FR_mt + SR * (UstR_mt - mtR)
        m0 = SL >= 0
        m1 = ~m0 & (Sstar >= 0)
        m2 = ~m0 & ~m1 & (SR >= 0)
        fh = np.where(m0, FL_h, np.where(m1, FstL_h,
                                          np.where(m2, FstR_h, FR_h)))
        fmn = np.where(m0, FL_mn, np.where(m1, FstL_mn,
                                            np.where(m2, FstR_mn, FR_mn)))
        fmt = np.where(m0, FL_mt, np.where(m1, FstL_mt,
                                            np.where(m2, FstR_mt, FR_mt)))
        return fh, fmn, fmt

    # ---- 逐胞右端项（minmod 重构 + HLLC 通量 + 强迫/拖曳 + 超粘性）----
    def _tend(self, H, U, V):
        gp = self.gp
        dx, dy = self.dx, self.dy
        nc = 1.0

        # —— x 方向：minmod 重构界面状态 → HLLC ——
        sH = self._slope_minmod(H); sU = self._slope_minmod(U)
        sV = self._slope_minmod(V)
        HLf = H + 0.5 * nc * sH; ULf = U + 0.5 * nc * sU
        VLf = V + 0.5 * nc * sV
        HRf = np.roll(H, -1, axis=1) - 0.5 * nc * np.roll(sH, -1, axis=1)
        URf = np.roll(U, -1, axis=1) - 0.5 * nc * np.roll(sU, -1, axis=1)
        VRf = np.roll(V, -1, axis=1) - 0.5 * nc * np.roll(sV, -1, axis=1)
        FxH, FxU, FxV = self._hllc(HLf, ULf, VLf, HRf, URf, VRf)
        dFxH = (FxH - np.roll(FxH, 1, axis=1)) / dx
        dFxU = (FxU - np.roll(FxU, 1, axis=1)) / dx
        dFxV = (FxV - np.roll(FxV, 1, axis=1)) / dx

        # —— y 方向：重构界面通量（normal=y），wall: Fy=0 ——
        sH = self._slope_minmod_y(H); sU = self._slope_minmod_y(U)
        sV = self._slope_minmod_y(V)
        HLf = H[:-1, :] + 0.5 * nc * sH[:-1, :]
        ULf = U[:-1, :] + 0.5 * nc * sU[:-1, :]
        VLf = V[:-1, :] + 0.5 * nc * sV[:-1, :]
        HRf = H[1:, :] - 0.5 * nc * sH[1:, :]
        URf = U[1:, :] - 0.5 * nc * sU[1:, :]
        VRf = V[1:, :] - 0.5 * nc * sV[1:, :]
        # normal=y ⇒ mn=V(质量通量), mt=U
        FyH, FyV, FyU = self._hllc(HLf, VLf, ULf, HRf, VRf, URf)
        dyH = np.empty_like(H); dyU = np.empty_like(U); dyV = np.empty_like(V)
        dyH[0] = FyH[0] / dy
        dyH[1:-1] = (FyH[1:] - FyH[:-1]) / dy
        dyH[-1] = -FyH[-1] / dy
        dyU[0] = FyU[0] / dy
        dyU[1:-1] = (FyU[1:] - FyU[:-1]) / dy
        dyU[-1] = -FyU[-1] / dy
        dyV[0] = FyV[0] / dy
        dyV[1:-1] = (FyV[1:] - FyV[:-1]) / dy
        dyV[-1] = -FyV[-1] / dy

        dH = -dFxH - dyH - (H - self.h_eq) / self.tau_rad
        dU = -dFxU - dyU + self.f * V - U / self.tau_drag
        dV = -dFxV - dyV - self.f * U - V / self.tau_drag

        if self.nu4 > 0:
            dH = dH - self.nu4 * self._bih(H)
            dU = dU - self.nu4 * self._bih(U)
            dV = dV - self.nu4 * self._bih(V)
        return dH, dU, dV

    def _bih(self, F):
        """中心差分双调和 ∇⁴（周期 x、通道 y，墙处用 Neumann 假胞）。"""
        nx, ny = self.nx, self.ny
        dx2 = self.dx ** 2; dy2 = self.dy ** 2
        Fp = np.roll(F, -1, axis=1); Fm = np.roll(F, 1, axis=1)
        lapx = (Fp - 2 * F + Fm) / dx2
        # y（墙外镜像）
        Fup = np.empty_like(F); Fdn = np.empty_like(F)
        Fdn[0] = F[0]; Fdn[1:] = F[:-1]
        Fup[-1] = F[-1]; Fup[:-1] = F[1:]
        lapy = (Fup - 2 * F + Fdn) / dy2
        L = lapx + lapy
        Lp = np.roll(L, -1, axis=1); Lm = np.roll(L, 1, axis=1)
        bip = (Lp - 2 * L + Lm) / dx2
        Lupn = np.empty_like(L); Ldnn = np.empty_like(L)
        Ldnn[0] = L[0]; Ldnn[1:] = L[:-1]
        Lupn[-1] = L[-1]; Lupn[:-1] = L[1:]
        biy = (Lupn - 2 * L + Ldnn) / dy2
        return bip + biy

    def cfl_timestep(self):
        H = self.H
        c = np.sqrt(self.c2)
        u = np.abs(self.U / H)
        v = np.abs(self.V / H)
        cmax = np.sqrt(self.gp * (H.max() if H.size else self.H0))
        umax = float(np.max(u + v)) + cmax
        dxm = min(self.dx, self.dy)
        dt_adv = self.cfl * dxm / (umax + 1e-9)
        dt_hyp = 0.1 * min(self.dx, self.dy) ** 4 / max(self.nu4, 1e-30)
        dt_force = 0.1 * self.tau_rad
        return min(dt_adv, dt_hyp, dt_force)

    def energy_density(self, H, U, V):
        ke = np.mean(0.5 * (U * U + V * V) / H)
        pe = np.mean(0.5 * self.gp * (H - self.H0) ** 2 / self.H0)
        return ke + pe

    def integrate(self, t_end_s: float) -> dict:
        H, U, V = self.H, self.U, self.V
        t = 0.0
        n = 0
        t_hist = [t]
        e_hist = [float(self.energy_density(H, U, V))]
        m_hist = [float(np.mean(H))]
        dt0 = self.cfl_timestep()
        while t < t_end_s - 1e-9:
            dt = self.cfl_timestep()
            dt = min(dt, t_end_s - t)
            dH0, dU0, dV0 = self._tend(H, U, V)
            Hh = H + 0.5 * dt * dH0
            Uh = U + 0.5 * dt * dU0
            Vh = V + 0.5 * dt * dV0
            dH1, dU1, dV1 = self._tend(Hh, Uh, Vh)
            H += dt * dH1
            U += dt * dU1
            V += dt * dV1
            t += dt
            n += 1
            if not (np.isfinite(H).all() and np.isfinite(U).all()
                    and np.isfinite(V).all() and (H > 0).all()):
                raise FloatingPointError(
                    f"非线性FVM发散 @ t={t:.3e}s n={n}")
            if n > 2_000_000:
                raise RuntimeError("积分步骤异常过多")
            t_hist.append(t)
            e_hist.append(float(self.energy_density(H, U, V)))
            m_hist.append(float(np.mean(H)))

        t_arr = np.asarray(t_hist); e_arr = np.asarray(e_hist)
        m_arr = np.asarray(m_hist)
        w = self.tau_rad
        d = np.abs(t_arr - t_arr[-1])
        ma = d < w
        mb = (d > w) & (d < 2 * w)
        E_late = float(np.mean(e_arr[ma])); E_prior = float(np.mean(e_arr[mb]))
        rel_energy_drift = (E_late - E_prior) / E_prior if E_prior > 1e-30 else 0.0
        mass_late = float(np.mean(m_arr[ma]))
        mass_drift = (mass_late - self.H0) / self.H0
        return {"name": self.name, "t_end_s": t, "steps": n,
                "dt_s": dt0, "CFL": self.cfl, "mass_drift": mass_drift,
                "rel_energy_drift": rel_energy_drift, "t_sub": self.t_sub,
                "nu4": self.nu4}

    def temperature(self):
        return self.t_sub * (1.0 + (self.H / self.H0 - 1.0) / self.amp)

    def analyze(self) -> dict:
        T = self.temperature()
        U = self.U; H = self.H
        ufield = U / H
        nx, ny = self.nx, self.ny
        R = self.sys.planet_radius_m
        deg = 180.0 / PI
        eq_rows = [j for j in range(ny) if abs(self.yp[j]) < (15.0 * PI / 180.0) * R]
        day_idx = [i for i in range(nx) if abs(self.lons_rad[i]) < PI / 2.0]
        night_idx = [i for i in range(nx) if abs(self.lons_rad[i]) >= PI / 2.0]
        dayT = float(np.mean([[T[j, i] for i in day_idx] for j in eq_rows]))
        nightT = float(np.mean([[T[j, i] for i in night_idx] for j in eq_rows]))
        T_merid = np.mean(T[eq_rows, :], axis=0)
        imax = int(np.argmax(T_merid))
        offset_deg = self.lons_rad[imax] * deg
        u_zonal = np.mean(ufield[eq_rows, :], axis=1)
        jet_max = float(np.max(u_zonal))
        return {
            "day_mean_K": round(dayT, 1), "night_mean_K": round(nightT, 1),
            "day_night_delta_T_K": round(dayT - nightT, 1),
            "hot_spot_offset_deg": round(offset_deg, 1),
            "equatorial_jet_max_ms": round(jet_max, 1),
        }


def run_scenario2(name: str = "WASP-43b", nx: int = 64, ny: int = 32,
                  t_end_s: float = 2.0e6, tau_rad: float | None = None,
                  tau_drag: float | None = None, nu4: float | None = None,
                  cfl: float | None = None, amp: float | None = None,
                  night_frac: float | None = None,
                  dealias_factor: float = 2.0 / 3.0,
                  verbose: bool = False) -> dict:
    """便捷入口：取系统→建立非线性谱求解器→积分→返回终态诊断。

    返回 dict 含：mass_drift、rel_energy_drift、day_night_delta_T_K、
    hot_spot_offset_deg（东>0）、equatorial_jet_max_ms（东向为正，m/s）。
    """
    sw = NonlinearSWChannelsFVM(name, nx=nx, ny=ny, tau_rad=tau_rad,
                                tau_drag=tau_drag, nu4=nu4, cfl=cfl, amp=amp,
                                night_frac=night_frac)
    r = sw.integrate(t_end_s)
    a = sw.analyze()
    r.update(a)
    if verbose:
        print(f"[{name}] nx{nx}xny{ny} dt={r['dt_s']:.1f}s steps={r['steps']} "
              f"CFL={r['CFL']}")
        print(f"  mass_drift={r['mass_drift']:+.3e} "
              f"rel_energy_drift={r['rel_energy_drift']:+.3e} "
              f"ΔT_day_night={r['day_night_delta_T_K']}K "
              f"offset={r['hot_spot_offset_deg']}° "
              f"equatorial_jet_max={r['equatorial_jet_max_ms']} m/s")
    return r


if __name__ == "__main__":
    res = run_scenario2("WASP-43b", nx=64, ny=32, t_end_s=2.0e6, verbose=True)
    print("done")