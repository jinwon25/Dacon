"""encoders.py — 새 decorrelated 인코더 변종 (proven 패턴: MLP→GRU 교체가 frenet_gru 다양성을 만듦).
공유 ODE 헤드(AccelField + RK4, GRUODEModel과 동일) + 새 인코더(Transformer / LRU / TCN).

리서치 근거:
- Transformer(11스텝 짧아 MLP로 degenerate 경향 → 정확도보단 decorrelation 목적, CLS pooling, pre-LN, dropout↑)
- LRU(선형 복소 diagonal recurrence — GRU 게이트/ODE 연속장과 근본적으로 다른 inductive bias; sysid-pytorch-lru)
- TCN(dilated causal conv = FIR 필터, recurrence/ODE와 직교)

인터페이스: model(seq(B,T,C), scal, init_vel, speed) -> local displacement.  model._last_accels 로 accel reg 노출.
v131 run_kfold와 호환 (encoder != 'mlp' 이면 seq를 (B,T,C)로 그대로 받음).
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v120_neural_ode import ResBlock, AccelField


class _ODEHead(nn.Module):
    """latent -> RK4 적분(80ms) -> local displacement. GRUODEModel 헤드와 동일."""
    def __init__(self, latent_dim=64, hidden=64, dt_pred=0.080, n_steps=1):
        super().__init__()
        self.dt_pred = dt_pred; self.n_steps = n_steps
        self.accel_field = AccelField(latent_dim=latent_dim, hidden=hidden)
        self.learned_damping = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
        self.local_bias = nn.Parameter(torch.zeros(3))
        self.last_accels = []

    def _deriv(self, pos, vel, latent, speed):
        a = self.accel_field(pos, vel, latent, speed)
        return vel, -self.learned_damping * vel + a, a

    def _rk4(self, pos, vel, latent, speed, dt):
        dp1, dv1, a1 = self._deriv(pos, vel, latent, speed)
        dp2, dv2, a2 = self._deriv(pos + .5*dt*dp1, vel + .5*dt*dv1, latent, speed)
        dp3, dv3, a3 = self._deriv(pos + .5*dt*dp2, vel + .5*dt*dv2, latent, speed)
        dp4, dv4, a4 = self._deriv(pos + dt*dp3, vel + dt*dv3, latent, speed)
        np_ = pos + (dt/6)*(dp1+2*dp2+2*dp3+dp4)
        nv_ = vel + (dt/6)*(dv1+2*dv2+2*dv3+dv4)
        return np_, nv_, [a1, a2, a3, a4]

    def integrate(self, latent, init_vel, speed):
        pos = torch.zeros_like(init_vel); vel = init_vel
        dt = self.dt_pred / self.n_steps; accels = []
        for _ in range(self.n_steps):
            pos, vel, ac = self._rk4(pos, vel, latent, speed, dt); accels.extend(ac)
        self.last_accels = accels
        return pos + self.local_bias


class _Fuse(nn.Module):
    def __init__(self, latent_dim, scal_dim):
        super().__init__()
        self.scal_proj = nn.Sequential(nn.Linear(scal_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(2*latent_dim, latent_dim), nn.LayerNorm(latent_dim),
                                  nn.GELU(), ResBlock(latent_dim))
    def forward(self, h, scal):
        return self.fuse(torch.cat([h, self.scal_proj(scal)], dim=-1))


class TransformerODE(nn.Module):
    """11스텝 self-attention + CLS pooling -> latent -> ODE 헤드."""
    def __init__(self, seq_channels=9, scal_dim=40, latent_dim=64, hidden=64, n_steps=1,
                 nhead=4, nlayers=2, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Linear(seq_channels, latent_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.pos = nn.Parameter(torch.zeros(1, 16, latent_dim))  # 11 steps + CLS, 여유
        layer = nn.TransformerEncoderLayer(latent_dim, nhead, dim_feedforward=2*latent_dim,
                                           dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.fuse = _Fuse(latent_dim, scal_dim)
        self.head = _ODEHead(latent_dim, hidden, n_steps=n_steps)
        nn.init.trunc_normal_(self.cls, std=0.02); nn.init.trunc_normal_(self.pos, std=0.02)

    @property
    def _last_accels(self): return self.head.last_accels

    def forward(self, seq, scal, init_vel, speed):
        B, T, _ = seq.shape
        x = self.in_proj(seq)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1) + self.pos[:, :T+1]
        h = self.tr(x)[:, 0]                       # CLS token
        return self.head.integrate(self.fuse(h, scal), init_vel, speed)


class LRULayer(nn.Module):
    """Linear Recurrent Unit (Orvieto 2023): diagonal complex recurrence, stable ring init."""
    def __init__(self, d_in, d_state, r_min=0.4, r_max=0.95):
        super().__init__()
        import math
        u1 = torch.rand(d_state); u2 = torch.rand(d_state)
        nu = -0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2)   # |A|=exp(-nu)
        theta = u2 * (2 * math.pi)
        self.nu_log = nn.Parameter(torch.log(nu)); self.theta = nn.Parameter(theta)
        self.B_re = nn.Parameter(torch.randn(d_state, d_in) / (d_in ** 0.5))
        self.B_im = nn.Parameter(torch.randn(d_state, d_in) / (d_in ** 0.5))
        self.C_re = nn.Parameter(torch.randn(d_in, d_state) / (d_state ** 0.5))
        self.C_im = nn.Parameter(torch.randn(d_in, d_state) / (d_state ** 0.5))
        self.D = nn.Parameter(torch.randn(d_in))

    def forward(self, u):                          # u: (B,T,d_in)
        B, T, _ = u.shape
        amp = torch.exp(-torch.exp(self.nu_log))   # |A| in (0,1)
        a_re = amp * torch.cos(self.theta); a_im = amp * torch.sin(self.theta)
        bu_re = u @ self.B_re.t(); bu_im = u @ self.B_im.t()   # (B,T,d_state)
        s_re = torch.zeros(B, a_re.shape[0], device=u.device)
        s_im = torch.zeros_like(s_re); outs = []
        for t in range(T):
            ns_re = a_re * s_re - a_im * s_im + bu_re[:, t]
            ns_im = a_re * s_im + a_im * s_re + bu_im[:, t]
            s_re, s_im = ns_re, ns_im
            y = s_re @ self.C_re.t() - s_im @ self.C_im.t() + u[:, t] * self.D
            outs.append(y)
        return torch.stack(outs, dim=1)            # (B,T,d_in)


class LRUODE(nn.Module):
    """LRU 인코더(2층) -> last step -> latent -> ODE 헤드."""
    def __init__(self, seq_channels=9, scal_dim=40, latent_dim=64, hidden=64, n_steps=1, d_state=32):
        super().__init__()
        self.in_proj = nn.Linear(seq_channels, latent_dim)
        self.lru1 = LRULayer(latent_dim, d_state); self.ln1 = nn.LayerNorm(latent_dim)
        self.lru2 = LRULayer(latent_dim, d_state); self.ln2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU())
        self.fuse = _Fuse(latent_dim, scal_dim)
        self.head = _ODEHead(latent_dim, hidden, n_steps=n_steps)

    @property
    def _last_accels(self): return self.head.last_accels

    def forward(self, seq, scal, init_vel, speed):
        x = self.in_proj(seq)
        x = x + self.lru1(self.ln1(x)); x = x + self.mlp(self.lru2(self.ln2(x)))
        return self.head.integrate(self.fuse(x[:, -1], scal), init_vel, speed)


class TCNODE(nn.Module):
    """Dilated causal conv(FIR) 인코더 -> last step -> latent -> ODE 헤드."""
    def __init__(self, seq_channels=9, scal_dim=40, latent_dim=64, hidden=64, n_steps=1, dilations=(1, 2, 4)):
        super().__init__()
        self.in_proj = nn.Linear(seq_channels, latent_dim)
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.ModuleDict({
                "conv": nn.Conv1d(latent_dim, latent_dim, kernel_size=3, dilation=d, padding=d),
                "ln": nn.GroupNorm(1, latent_dim), "act": nn.GELU(), "drop": nn.Dropout(0.1)}))
        self.fuse = _Fuse(latent_dim, scal_dim)
        self.head = _ODEHead(latent_dim, hidden, n_steps=n_steps)

    @property
    def _last_accels(self): return self.head.last_accels

    def forward(self, seq, scal, init_vel, speed):
        x = self.in_proj(seq).transpose(1, 2)      # (B,latent,T)
        T = x.shape[-1]
        for blk in self.blocks:
            z = blk["conv"](x)[..., :T]            # causal trim
            x = x + blk["drop"](blk["act"](blk["ln"](z)))
        return self.head.integrate(self.fuse(x[:, :, -1], scal), init_vel, speed)


class iTransformerODE(nn.Module):
    """iTransformer: 각 채널(rel/vel/accel ×xyz=9)을 토큰으로, 시계열을 임베딩 → 채널 간 attention.
    시간축이 아닌 '변수 간' 결합을 모델링 → 시간기반 GRU/ODE/conv와 근본적으로 다른 inductive bias."""
    def __init__(self, seq_channels=9, scal_dim=40, latent_dim=64, hidden=64, n_steps=1,
                 nhead=4, nlayers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.LazyLinear(latent_dim)     # 각 채널의 T-시계열 -> 토큰 임베딩 (T는 첫 forward에서 추론)
        layer = nn.TransformerEncoderLayer(latent_dim, nhead, dim_feedforward=2*latent_dim,
                                           dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.pool = nn.Sequential(nn.Linear(seq_channels * latent_dim, latent_dim), nn.GELU())
        self.fuse = _Fuse(latent_dim, scal_dim)
        self.head = _ODEHead(latent_dim, hidden, n_steps=n_steps)

    @property
    def _last_accels(self): return self.head.last_accels

    def forward(self, seq, scal, init_vel, speed):
        B, T, C = seq.shape
        tok = self.embed(seq.transpose(1, 2))      # (B,C,T) -> (B,C,latent), 채널이 토큰
        h = self.tr(tok)                            # (B,C,latent) 채널 간 attention
        latent = self.pool(h.reshape(B, -1))
        return self.head.integrate(self.fuse(latent, scal), init_vel, speed)


class FlowODE(nn.Module):
    """Neural Flow(out-of-cluster): CV 베이스라인 + 직접 flow 잔차. RK4 적분도 CREE 회전도 아님 → 3번째 메커니즘.
    disp = gate*init_vel*T + flow_net(latent, init_vel, speed). 적분 inductive bias 제거 → frenet-RK4와 구조 직교 시도."""
    def __init__(self, seq_channels=9, scal_dim=40, latent_dim=64, hidden=64, n_steps=1, dt_pred=0.080):
        super().__init__()
        self.dt_pred = dt_pred
        self.gru = nn.GRU(seq_channels, latent_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fuse = _Fuse(latent_dim, scal_dim)
        self.flow = nn.Sequential(nn.Linear(latent_dim + 4, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                  ResBlock(hidden), nn.Linear(hidden, 3))
        self.vel_gate = nn.Parameter(torch.tensor(1.0))
        self._accels = []
        with torch.no_grad():  # flow 출력 0 근처에서 출발 (CV prior)
            nn.init.zeros_(self.flow[-1].weight); nn.init.zeros_(self.flow[-1].bias)

    @property
    def _last_accels(self): return self._accels

    def forward(self, seq, scal, init_vel, speed):
        h = self.gru(seq)[0][:, -1]
        latent = self.fuse(h, scal)
        resid = self.flow(torch.cat([latent, init_vel, speed.unsqueeze(-1)], dim=-1))
        disp = self.vel_gate * init_vel * self.dt_pred + resid
        self._accels = [resid]   # tiny accel-reg에 무해
        return disp


class SONODEModel(nn.Module):
    """SONODE(2nd-order): 초기속도 v0를 history에서 학습(관측 init_vel에 보정) → 다른 trajectory family.
    RK4 적분 유지하되 시작조건이 달라져 frenet-RK4와 decorrelate 시도 (research #2 pick)."""
    def __init__(self, seq_channels=9, scal_dim=40, latent_dim=64, hidden=64, n_steps=1):
        super().__init__()
        self.gru = nn.GRU(seq_channels, latent_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fuse = _Fuse(latent_dim, scal_dim)
        self.v0_net = nn.Sequential(nn.Linear(latent_dim + 4, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.head = _ODEHead(latent_dim, hidden, n_steps=n_steps)
        with torch.no_grad(): nn.init.zeros_(self.v0_net[-1].weight); nn.init.zeros_(self.v0_net[-1].bias)

    @property
    def _last_accels(self): return self.head.last_accels

    def forward(self, seq, scal, init_vel, speed):
        h = self.gru(seq)[0][:, -1]
        latent = self.fuse(h, scal)
        v0 = init_vel + self.v0_net(torch.cat([latent, init_vel, speed.unsqueeze(-1)], dim=-1))  # 학습된 v0 보정
        return self.head.integrate(latent, v0, speed)


ENCODER_REGISTRY = {"transformer": TransformerODE, "lru": LRUODE, "tcn": TCNODE,
                    "itransformer": iTransformerODE, "flow": FlowODE, "sonode": SONODEModel}
